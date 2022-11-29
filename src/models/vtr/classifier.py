from typing import Dict, Tuple

import torch
from torch import nn

from src.models.vtr.embedder import VisualEmbedder


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.pow(
        #     1e4, -torch.arange(0, hidden_size, 2).float() / hidden_size
        # )
        # pe = torch.zeros(max_len, 1, hidden_size)
        # pe[:,0,0::2] = torch.sin(position * div_term)
        # pe[:,0,1::2] = torch.cos(position * div_term)
        # self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class VisualToxicClassifier(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        kernel_size: int = 3,
        emb_size: int = 768,
        num_layers: int = 12,
        nhead: int = 12,
        out_channels: int = 32,
        dropout: float = 0,
    ):
        super().__init__()
        self.embedder = VisualEmbedder(
            height=height,
            width=width,
            kernel_size=kernel_size,
            emb_size=emb_size,
            out_channels=out_channels,
        )

        self.positional = PositionalEncoding(emb_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            dim_feedforward=3072,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(emb_size, 1)

    def forward(self, input_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedder(input_batch["slices"])  # batch_size, seq_len, emb_size

        embeddings = self.positional(embeddings)
        encoder_output = self.encoder(
            src=embeddings,
            src_key_padding_mask=input_batch["attention_mask"],
        )
        encoder_output = encoder_output.mean(dim=1)
        return self.classifier(encoder_output).squeeze(1)
