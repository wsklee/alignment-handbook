#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

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
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

from alignment.configs import ModelArguments
from alignment.preprocessing import preprocess_function

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

class ContrastiveTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Create input dicts
        sequence_1_inputs = {k: v for k, v in inputs.items() 
                           if not k.endswith('_pair') and k != 'labels'}
        sequence_2_inputs = {k.replace('_pair', ''): v for k, v in inputs.items() 
                           if k.endswith('_pair')}
        
        print("DEBUG - sequence_1_inputs keys:", sequence_1_inputs.keys())
        print("DEBUG - sequence_2_inputs keys:", sequence_2_inputs.keys())
        
        outputs_1 = model(**sequence_1_inputs)
        outputs_2 = model(**sequence_2_inputs)
        
        # Get embeddings for both sequences
        embeddings_1 = outputs_1.logits
        embeddings_2 = outputs_2.logits
        
        # Normalize embeddings
        embeddings_1 = torch.nn.functional.normalize(embeddings_1, dim=1)
        embeddings_2 = torch.nn.functional.normalize(embeddings_2, dim=1)
        
        # Compute similarity scores
        similarity = torch.sum(embeddings_1 * embeddings_2, dim=1)
        
        # Get labels
        labels = inputs["labels"].float()
        
        # Compute contrastive loss
        temperature = self.args.temperature
        loss = torch.mean((1 - labels) * torch.square(similarity) + 
                         labels * torch.square(torch.clamp(1 - similarity, min=0.0)))
        
        if return_outputs:
            return loss, {'logits': similarity}
        return loss

# Create a custom data collator that handles paired sequences
class ContrastiveDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.default_collator = DataCollatorWithPadding(tokenizer)
    
    def __call__(self, features):
        print("DEBUG - Raw features keys:", features[0].keys())  # More specific debug print
        
        # Separate features into first and second sequences
        batch_1 = []
        batch_2 = []
        for f in features:
            # First sequence
            item1 = {
                'input_ids': f['input_ids'],
                'attention_mask': f['attention_mask']
            }
            # Second sequence - add error handling
            try:
                item2 = {
                    'input_ids': f['input_ids_pair'],
                    'attention_mask': f['attention_mask_pair']
                }
            except KeyError as e:
                print(f"DEBUG - Missing key in features: {e}")
                print(f"DEBUG - Available keys: {f.keys()}")
                raise KeyError(f"Missing paired sequence keys. Expected 'input_ids_pair' and 'attention_mask_pair', got keys: {f.keys()}")
            
            batch_1.append(item1)
            batch_2.append(item2)
        
        # Collate each batch separately
        collated_1 = self.default_collator(batch_1)
        collated_2 = self.default_collator(batch_2)
        
        # Combine results
        final_batch = {}
        for k, v in collated_1.items():
            final_batch[k] = v
            final_batch[f"{k}_pair"] = collated_2[k]
        
        # Add labels
        if features and "labels" in features[0]:
            final_batch["labels"] = torch.tensor([f["labels"] for f in features])
        
        print("DEBUG - Final batch keys:", final_batch.keys())
        return final_batch

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

    # Load datasets
    raw_datasets = {}
    for dataset_name, weight in data_args.dataset_mixer.items():
        if dataset_name == "imdb":
            dataset = load_dataset("imdb")
        elif dataset_name == "sst2":
            dataset = load_dataset("glue", "sst2")
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


    # Process datasets
    processed_datasets = {}
    for name, dataset in raw_datasets.items():
        processed_datasets[name] = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, data_args.max_seq_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=dataset["train"].column_names,
            desc=f"Processing {name} dataset",
        )
        
        # Debug print right after processing
        print(f"DEBUG - Processed dataset keys for {name}:", 
              processed_datasets[name]["train"].features.keys())
        print(f"DEBUG - Processed first item keys for {name}:", 
              next(iter(processed_datasets[name]["train"])).keys())

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
    # Debug print after concatenation
    print("DEBUG - After concat keys:", train_dataset.features.keys())
    print("DEBUG - After concat first item keys:", next(iter(train_dataset)).keys())

    eval_dataset = datasets.concatenate_datasets(eval_datasets)

    # Data collator
    data_collator = ContrastiveDataCollator(tokenizer)

    # Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # For contrastive learning, predictions are similarity scores
        # Convert to binary predictions based on threshold (e.g., 0.5)
        binary_predictions = (predictions > 0.5).astype(int)
        return metric.compute(predictions=binary_predictions, references=labels)

    # Initialize callbacks list
    callbacks = []
    if training_args.early_stopping_patience > 0:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
            early_stopping_threshold=training_args.early_stopping_threshold,
        )
        callbacks.append(early_stopping_callback)

    # Before creating trainer
    print("DEBUG - Train dataset features:", train_dataset.features)
    print("DEBUG - Sample from dataset:", next(iter(train_dataset)))

    # Initialize ContrastiveTrainer
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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