from typing import Dict

import torch
from torch import nn
from transformers import BertConfig, BertModel


class EncoderForSequenceLabeling(nn.Module):
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
            num_layers=num_layers,
            dropout=dropout,
        )
        self.bert = BertModel(model_config)
        self.word_classifier = nn.Linear(hidden_size, 1)

    def forward(self, batch: Dict[str, torch.Tensor]):
        attention_mask = batch["attention_mask"]

        bert_output = self.bert(input_ids=batch["input_ids"], attention_mask=attention_mask)

        masked_output = bert_output["last_hidden_state"] * attention_mask[:, :, None]
        batch_size, seq_len, hidden_size = masked_output.shape
        word_len = int(batch["max_word_len"].item())
        word_split_output = masked_output.view(batch_size, seq_len // word_len, word_len, hidden_size)
        tokens_count_in_each_word = attention_mask.view(batch_size, seq_len // word_len, word_len).sum(dim=2)
        word_embeddings = (
            torch.mean(word_split_output, 2) / torch.max(torch.tensor(1), tokens_count_in_each_word)[:, :, None]
        )
        return self.word_classifier(word_embeddings).squeeze(2)
