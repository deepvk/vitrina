from typing import Dict

import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30_000,
        max_position_embeddings: int = 514,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 1,
        type_vocab_size: int = 1,
        num_layers: int = 1,
        dropout=0.0,
    ):
        super().__init__()
        model_config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            num_labels=num_layers,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.model = BertForSequenceClassification(model_config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        bert_output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )

        return bert_output["logits"].squeeze(1)
