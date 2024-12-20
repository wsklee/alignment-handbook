#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

from alignment.configs import ModelArguments
from alignment.model_utils import get_checkpoint

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """Arguments for dataset configuration"""
    dataset_mixer: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of dataset names and their mixing weights"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum sequence length to use"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of workers for preprocessing"}
    )

@dataclass
class SFTConfig(TrainingArguments):
    """Arguments for SFT training that extend the base TrainingArguments"""
    peft_lora_r: int = field(
        default=8,
        metadata={"help": "Lora R dimension"}
    )
    peft_lora_alpha: int = field(
        default=16,
        metadata={"help": "Lora alpha"}
    )
    peft_lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Lora dropout"}
    )
    peft_lora_modules: List[str] = field(
        default_factory=lambda: ["q_lin", "k_lin", "v_lin", "out_lin", "ffn.lin1", "ffn.lin2"],
        metadata={"help": "List of module names to apply Lora to"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    set_seed(training_args.seed)

    # Load datasets
    raw_datasets = {}
    for dataset_name, weight in data_args.dataset_mixer.items():
        if dataset_name == "imdb":
            dataset = load_dataset("imdb")
        elif dataset_name == "sst2":
            dataset = load_dataset("glue", "sst2")
        elif dataset_name == "yelp_polarity":
            dataset = load_dataset("yelp_polarity")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets: imdb, sst2, yelp")
        raw_datasets[dataset_name] = dataset

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        num_labels=2,  # Binary classification
    )

    # Configure LoRA if enabled
    if hasattr(training_args, "use_peft") and training_args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=training_args.peft_lora_r,
            lora_alpha=training_args.peft_lora_alpha,
            lora_dropout=training_args.peft_lora_dropout,
            target_modules=training_args.peft_lora_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Preprocessing function
    def preprocess_function(examples):
        # For SST-2 the text field is "sentence", for IMDB it's "text"
        texts = examples.get("sentence", examples.get("text", []))
        result = tokenizer(
            texts,
            padding=False,
            max_length=data_args.max_seq_length,
            truncation=True,
        )
        
        # Add labels
        result["labels"] = examples["label"]
        return result

    # Process datasets
    processed_datasets = {}
    for name, dataset in raw_datasets.items():
        processed_datasets[name] = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=dataset["train"].column_names,
            desc=f"Processing {name} dataset",
        )

    # Combine datasets
    train_datasets = []
    eval_datasets = []
    for name, weight in data_args.dataset_mixer.items():
        train_datasets.append(processed_datasets[name]["train"])
        if "validation" in processed_datasets[name]:
            eval_datasets.append(processed_datasets[name]["validation"])
        else:
            eval_datasets.append(processed_datasets[name]["test"])

    train_dataset = datasets.concatenate_datasets(train_datasets)
    eval_dataset = datasets.concatenate_datasets(eval_datasets)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()