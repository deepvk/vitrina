import torch
from loguru import logger
from torch import nn
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from src.models.vtr.embedder import VisualEmbedder
from src.models.vtr.ocr import OCRHead

from src.utils.common import char2int


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
        max_position_embeddings: int = 512,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 1,
        dropout: float = 0.0,
        out_channels: int = 32,
        ocr_flag: bool = True,
    ):
        super().__init__()
        logger.info(f"Initializing VTR classifier | hidden size: {hidden_size}, # layers: {num_layers}")

        self.embedder = VisualEmbedder(
            height=height,
            width=width,
            kernel_size=kernel_size,
            emb_size=hidden_size,
            out_channels=out_channels,
        )

        self.positional = PositionalEncoding(hidden_size, dropout, max_len=max_position_embeddings)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            dim_feedforward=hidden_size * 4,
            nhead=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)
        self.ocr = OCRHead(input_size=256, hidden_size=256, num_layers=2, num_classes=60)
        self.ocr_flag = ocr_flag

    def forward(self, input_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        embeddings, conv = self.embedder(input_batch["slices"])  # batch_size, seq_len, emb_size
        embeddings = self.positional(embeddings)

        # If a BoolTensor is provided, the positions with the value of True will be ignored
        # while the position with the value of False will be unchanged.
        attn_mask = ~(input_batch["attention_mask"].bool())
        encoder_output = self.encoder(src=embeddings, src_key_padding_mask=attn_mask)  # batch_size, seq_len, emb_size

        encoder_output = encoder_output.mean(dim=1)  # batch_size, emb_size
        encoder_output = self.norm(encoder_output)  # batch_size, emb_size
        result = self.classifier(encoder_output).squeeze(1)  # batch_size

        # OCR
        if self.ocr_flag:
            criterion = CTCLoss(reduction="sum", zero_infinity=True)

            texts = list("".join(np.concatenate(input_batch["texts"]).flatten()))
            targets = char2int(texts)

            get_len = np.vectorize(len)
            target_lengths = pad_sequence(
                [torch.from_numpy(get_len(arr)) for arr in input_batch["texts"]], batch_first=True, padding_value=0
            )

            logits = self.ocr(conv)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            input_lengths = torch.LongTensor([log_probs.shape[0]] * log_probs.shape[1])

            ctc_loss = criterion(log_probs, targets, input_lengths, target_lengths)
            ctc_loss /= conv.shape[0]

            return result, ctc_loss

        else:
            return result
