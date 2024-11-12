import torch
from transformers import DistilBertModel, DistilBertPreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ContrastiveOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class DistilBertCLModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.init_weights()

    def get_embedding(self, input_ids, attention_mask):
        # print(f"DEBUG - get_embedding input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        embeddings = outputs.last_hidden_state[:, 0]
        #print(f"DEBUG - embedding output shape: {embeddings.shape}")
        return embeddings

    def forward(
        self,
        sent0_input_ids=None,
        sent0_attention_mask=None,
        sent1_input_ids=None,
        sent1_attention_mask=None,
        hard_neg_input_ids=None,
        hard_neg_attention_mask=None,
        return_dict=True,
    ):
        print("\nDEBUG - DistilBertCLModel forward pass")
        # print(f"DEBUG - Input shapes:")
        # print(f"  sent0: {sent0_input_ids.shape if sent0_input_ids is not None else None}")
        # print(f"  sent1: {sent1_input_ids.shape if sent1_input_ids is not None else None}")
        # print(f"  hard_neg: {hard_neg_input_ids.shape if hard_neg_input_ids is not None else None}")

        sent0_embed = self.get_embedding(sent0_input_ids, sent0_attention_mask)
        sent1_embed = self.get_embedding(sent1_input_ids, sent1_attention_mask)
        hard_neg_embed = self.get_embedding(hard_neg_input_ids, hard_neg_attention_mask)

        print(f"DEBUG - Output embedding shapes:")
        # print(f"  sent0: {sent0_embed.shape}")
        # print(f"  sent1: {sent1_embed.shape}")
        # print(f"  hard_neg: {hard_neg_embed.shape}")

        if not return_dict:
            return (sent0_embed, sent1_embed, hard_neg_embed)

        return ContrastiveOutput(
            embeddings=(sent0_embed, sent1_embed, hard_neg_embed)
        )