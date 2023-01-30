import torch
from loguru import logger
from torch import nn

from src.models.vtr.embedder import VisualEmbedder
from src.models.vtr.ocr import OCRHead


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
        num_classes: int = 2,
        dropout: float = 0.0,
        out_channels: int = 32,
        ocr_flag: bool = True,
        hidden_size_ocr: int = 256,
        num_layers_ocr: int = 2,
        num_classes_ocr: int = 44,
    ):
        super().__init__()
        logger.info(f"Initializing VTR classifier | hidden size: {hidden_size}, # layers: {num_layers}")
        if ocr_flag:
            logger.info(
                f"OCR parameters: hidden size: {hidden_size_ocr}, # layers: {num_layers_ocr}, # classes: {num_classes_ocr}"
            )

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
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
        self.ocr = OCRHead(
            input_size=out_channels * (height // 2**3),
            hidden_size=hidden_size_ocr,
            num_layers=num_layers_ocr,
            num_classes=num_classes_ocr,
        )
        self.ocr_flag = ocr_flag

    def forward(self, input_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings, conv = self.embedder(input_batch["slices"])  # batch_size, seq_len, emb_size
        embeddings = self.positional(embeddings)

        # If a BoolTensor is provided, the positions with the value of True will be ignored
        # while the position with the value of False will be unchanged.
        attn_mask = ~(input_batch["attention_mask"].bool())
        encoder_output = self.encoder(src=embeddings, src_key_padding_mask=attn_mask)  # batch_size, seq_len, emb_size

        encoder_output = encoder_output.mean(dim=1)  # batch_size, emb_size
        encoder_output = self.norm(encoder_output)  # batch_size, emb_size
        logits = self.classifier(encoder_output)  # batch_size, num_classes

        result = {"logits": logits}

        # OCR
        if self.ocr_flag:
            result["ocr_logits"] = self.ocr(conv)

        return result
