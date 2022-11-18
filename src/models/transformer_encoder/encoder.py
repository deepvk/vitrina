from typing import Dict

import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification


class Encoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        model_config = BertConfig(
            vocab_size=30_000,
            max_position_embeddings=514,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=1,
            type_vocab_size=1,
            num_labels=1,
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
