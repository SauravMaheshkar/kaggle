"""Custom Model Code"""
from typing import Dict

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

__all__ = ["FeedBackModel"]


class MeanPooling(nn.Module):
    """Mean Pooling Utility Class"""

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a forward pass through the model

        :param last_hidden_state: last hidden state from the base model
        :type last_hidden_state: torch.Tensor
        :param attention_mask: Attention Mask from the base model
        :type attention_mask: torch.Tensor
        :return: Mean Embeddings
        :rtype: torch.Tensor
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeedBackModel(nn.Module):
    """Model Class"""

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(cfg["model_name"])
        self.config.update(
            {
                "hidden_dropout_prob": 0.0,
                "add_pooling_layer": False,
                "hidden_act": "swish",
                "gradient_checkpointing": True,
            }
        )
        self.model = AutoModel.from_config(self.config)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        self.drop = nn.Dropout(p=cfg["classifier_dropout"])
        self.pooler = MeanPooling()
        self.output_layer = nn.Linear(self.config.hidden_size, cfg["num_classes"])

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass through the model

        :param ids: ids from the Dataset
        :type ids: torch.Tensor
        :param mask: mask from the Dataset
        :type mask: torch.Tensor
        :return: Output logits
        :rtype: torch.Tensor
        """
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.layer_norm(out)
        out = self.drop(out)
        outputs = self.output_layer(out)
        return outputs
