from transformers import BertConfig, BertForSequenceClassification
import torch
from torch import nn
from loguru import logger

from src.models.vtr.ocr import OCRHead
from src.utils.common import PositionalEncoding


class SequenceClassifier(nn.Module):
    def __init__(self, config, embedder, max_position_embeddings, ocr: OCRHead = None):
        super().__init__()

        logger.info(
            f"Initializing vanilla BERT classifier | hidden size: {config.emb_size}, " f"# layers: {config.num_layers}"
        )

        model_config = BertConfig(
            hidden_size=config.emb_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.n_head,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            num_classes=config.num_classes,
        )
        self.backbone = BertForSequenceClassification(model_config)
        self.embedder = embedder
        self.ocr = ocr
        self.positional = PositionalEncoding(config.emb_size, config.dropout, max_position_embeddings)

    def forward(self, input_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        output = self.embedder(input_batch)  # batch_size, seq_len, emb_size
        output["embeddings"] = self.positional(output["embeddings"])
        result = self.backbone(inputs_embeds=output["embeddings"])  # batch_size, num_classes

        if self.ocr:
            result["ocr_logits"] = self.ocr(output["ocr_embeddings"])

        return result
