from transformers import BertConfig, BertForSequenceClassification
import torch
from torch import nn
from loguru import logger

from src.models.vtr.ocr import OCRHead
from src.utils.common import PositionalEncoding


class SequenceClassifier(nn.Module):
    def __init__(self, config, embedder, ocr: OCRHead = None):
        super().__init__()

        logger.info(
            f"Initializing vanilla BERT classifier | hidden size: {config['hidden_size']}, "
            f"# layers: {config['num_hidden_layers']}")

        model_config = BertConfig(**config)
        self.backbone = BertForSequenceClassification(model_config)
        self.embedder = embedder
        self.ocr = ocr
        self.positional = PositionalEncoding(
            config["hidden_size"],
            config["hidden_dropout_prob"],
            config["max_position_embeddings"]
        )

    def forward(self, input_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}
        output = self.embedder(input_batch)  # batch_size, seq_len, emb_size
        output["embeddings"] = self.positional(output["embeddings"])
        if self.ocr:
            result["ocr_logits"] = self.ocr(output["ocr_embeddings"])

        logits = self.backbone(inputs_embeds=output["embeddings"])  # batch_size, num_classes
        result["logits"] = logits

        return result
