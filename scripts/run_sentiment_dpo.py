#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate

from alignment.configs import ModelArguments
from alignment.model_utils import get_checkpoint, get_tokenizer

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
class DPOConfig(TrainingArguments):
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter for DPO loss"}
    )
    reference_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model"}
    )

class SentimentDPOTrainer(Trainer):
    def __init__(self, reference_model: PreTrainedModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.reference_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # Get logits from current model
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get logits from reference model
        with torch.no_grad():
            ref_outputs = self.reference_model(**inputs)
            ref_logits = ref_outputs.logits
            
        # Compute DPO loss
        policy_logprobs = torch.log_softmax(logits, dim=-1)
        ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
        
        # Get true labels
        labels = inputs["labels"]
        
        # Compute loss using preference learning
        loss = -torch.mean(
            labels * (policy_logprobs[:, 1] - ref_logprobs[:, 1]) * self.args.beta
        )
        
        return (loss, outputs) if return_outputs else loss

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DPOConfig))
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
        raw_datasets[dataset_name] = dataset

    # Load tokenizer & models
    tokenizer = get_tokenizer(model_args)
    
    # Load policy model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,
        revision=model_args.model_revision,
    )
    
    # Load reference model
    reference_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reference_model or model_args.model_name_or_path,
        num_labels=2,
    )

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
        if "label" in examples:
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

    # Combine datasets according to mixing weights
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

    # Initialize DPO Trainer
    trainer = SentimentDPOTrainer(
        model=model,
        reference_model=reference_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = get_checkpoint(training_args)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
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