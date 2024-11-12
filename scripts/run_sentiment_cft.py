#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

from alignment.configs import ModelArguments

from alignment.cft.contrastive_trainer import ContrastiveTrainer
from alignment.cft.imdb_preprocess import IMDBPreprocess

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
class CFTConfig(TrainingArguments):
    """Arguments for CFT training that extend the base TrainingArguments"""
    temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for contrastive loss"}
    )
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
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Number of epochs to wait before early stopping"}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "How much the metric must improve to be considered as improved"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CFTConfig))
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

    # Load model
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True,
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

    # Load and preprocess IMDB dataset
    imdb_preprocessor = IMDBPreprocess(model_name=model_args.model_name_or_path)
    train_dataset = imdb_preprocessor.ds
    # Split dataset for evaluation (10% for eval)
    train_test_split = train_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Initialize callbacks list
    callbacks = []
    if training_args.early_stopping_patience > 0:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
            early_stopping_threshold=training_args.early_stopping_threshold,
        )
        callbacks.append(early_stopping_callback)


    # Initialize ContrastiveTrainer
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    # After creating trainer
    print("DEBUG - Trainer dataloader first batch:", 
          next(iter(trainer.get_train_dataloader())).keys())

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