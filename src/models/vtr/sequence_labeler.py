from typing import Dict

import torch
from torch import nn

from src.models.vtr.embedder import VisualEmbedderSL


class VisualTextSequenceLabeler(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        kernel_size: int = 3,
        emb_size: int = 768,
        num_layers: int = 1,
        out_channels: int = 32,
        nhead: int = 12,
        dropout: float = 0,
    ):
        super().__init__()
        self.embedder = VisualEmbedderSL(
            height=height,
            width=width,
            kernel_size=kernel_size,
            emb_size=emb_size,
            out_channels=out_channels,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            dim_feedforward=3072,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.class_predictor = nn.Linear(emb_size, 1)

    def forward(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = self.embedder(input_batch)

        decoder_output = self.encoder(
            src=embeddings,
            src_key_padding_mask=input_batch["words_mask"],
        )
        batch_size, max_seq_len, emb_size = decoder_output.shape
        over_all_batch = decoder_output.reshape(batch_size * max_seq_len, emb_size)
        class_logits = self.class_predictor(over_all_batch)
        return class_logits.view(batch_size, max_seq_len)
