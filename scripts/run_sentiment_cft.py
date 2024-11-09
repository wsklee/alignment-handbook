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
import torch.nn.functional as F
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
)
import evaluate

from alignment.configs import ModelArguments
from alignment.model_utils import get_checkpoint

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
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
    temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature parameter for contrastive loss"}
    )

class SentimentCFTTrainer(Trainer):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        # Get embeddings and labels
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, 0, :]  # Use [CLS] token embedding
        labels = inputs.pop("labels")
        
        # Normalize embeddings
        embeddings = F.normalize(hidden_states, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create masks for positive and negative pairs
        batch_size = labels.shape[0]
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_positives = labels_matrix.float() - torch.eye(batch_size, device=labels.device)
        mask_negatives = ~labels_matrix
        
        # Compute contrastive loss
        exp_similarities = torch.exp(similarity_matrix)
        positives = torch.sum(exp_similarities * mask_positives, dim=1)
        negatives = torch.sum(exp_similarities * mask_negatives, dim=1)
        loss = -torch.log(positives / (positives + negatives)).mean()
        
        if return_outputs:
            return loss, outputs
        return loss

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

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        num_labels=2,
    )

    # Preprocessing function
    def preprocess_function(examples):
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

    # Process and combine datasets
    processed_datasets = {}
    for name, dataset in raw_datasets.items():
        processed_datasets[name] = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=dataset["train"].column_names,
            desc=f"Processing {name} dataset",
        )

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

    # Data collator and metrics
    data_collator = DataCollatorWithPadding(tokenizer)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Initialize trainer
    trainer = SentimentCFTTrainer(
        temperature=training_args.temperature,
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