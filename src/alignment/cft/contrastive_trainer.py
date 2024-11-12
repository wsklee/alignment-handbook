from .loss import InfoNCE
from transformers import Trainer
import torch
import torch.nn.functional as F
from typing import Optional, List
from transformers.trainer_utils import EvalLoopOutput

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # If compute_metrics wasn't passed, use our implementation
        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = self.compute_metrics
        
        super().__init__(*args, **kwargs)
        temperature = kwargs.get("args").temperature
        self.info_nce = InfoNCE(temperature=temperature,
                                device=self.accelerator.device)
        print(f"DEBUG - Initialized ContrastiveTrainer with temperature={temperature}")
    
    def encode(self, model, x):
        
        inputs = {
            'input_ids': x['input_ids'].to(model.device),
            'attention_mask': x['attention_mask'].to(model.device)
        }
                
        outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        print("\nDEBUG - ContrastiveTrainer compute_loss")
        print("DEBUG - Input keys:", inputs.keys())
        print("DEBUG - Input shapes:", {k: v.shape for k, v in inputs.items()})
        print("DEBUG - Input device:", inputs['sent0_input_ids'].device)
        
        outputs = model(
            sent0_input_ids=inputs['sent0_input_ids'],
            sent0_attention_mask=inputs['sent0_attention_mask'],
            sent1_input_ids=inputs['sent1_input_ids'],
            sent1_attention_mask=inputs['sent1_attention_mask'],
            hard_neg_input_ids=inputs['hard_neg_input_ids'],
            hard_neg_attention_mask=inputs['hard_neg_attention_mask'],
        )
        
        sent0_embed, sent1_embed, hard_neg_embed = outputs.embeddings
        print("\nDEBUG - Embedding shapes:")
        print(f"  sent0: {sent0_embed.shape}")
        print(f"  sent1: {sent1_embed.shape}")
        print(f"  hard_neg: {hard_neg_embed.shape}")
        
        loss = self.info_nce(sent0_embed, sent1_embed, hard_neg_embed)
        print(f"DEBUG - Loss value: {loss.item()}")
        
        return loss
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics from the embeddings dictionary
        """
        # Unpack embeddings from the combined dictionary
        anchor_embeds = eval_pred['anchor']
        positive_embeds = eval_pred['positive']
        negative_embeds = eval_pred['negative']
        
        # Calculate similarities
        pos_cos_sim = F.cosine_similarity(anchor_embeds, positive_embeds)
        neg_cos_sim = F.cosine_similarity(anchor_embeds, negative_embeds)
        
        # Calculate metrics
        accuracy = (pos_cos_sim > neg_cos_sim).float().mean()
        
        metrics = {
            "eval_accuracy": accuracy.item(),
            "eval_pos_similarity": pos_cos_sim.mean().item(),
            "eval_neg_similarity": neg_cos_sim.mean().item(),
        }
        
        return metrics
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Custom evaluation loop for contrastive learning.
        Handles paired inputs and computes similarity-based metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_embeddings = []
        
        for step, inputs in enumerate(dataloader):
            with torch.no_grad():
                # Compute loss
                loss = self.compute_loss(self.model, inputs)
                total_loss += loss.item()
                
                # Get embeddings
                outputs = self.model(**inputs)
                sent0_embed, sent1_embed, hard_neg_embed = outputs.embeddings
                
                # Store embeddings for metric computation
                all_embeddings.append({
                    'anchor': sent0_embed,
                    'positive': sent1_embed,
                    'negative': hard_neg_embed
                })
        
        # Concatenate all embeddings
        combined_embeddings = {
            k: torch.cat([batch[k] for batch in all_embeddings])
            for k in ['anchor', 'positive', 'negative']
        }
        
        # Compute metrics
        metrics = self.compute_metrics(combined_embeddings)
        metrics[f"{metric_key_prefix}_loss"] = total_loss / len(dataloader)
        
        return EvalLoopOutput(
            predictions=combined_embeddings,
            label_ids=None,
            metrics=metrics,
            num_samples=len(dataloader.dataset)
        )