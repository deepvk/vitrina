from transformers import BertConfig, BertForSequenceClassification
import torch
from torch import nn
from loguru import logger
from torch.nn import CTCLoss

from src.models.vtr.ocr import OCRHead
from src.utils.common import PositionalEncoding, compute_ctc_loss
from src.models.embedders.vtr import VTREmbedder


class SequenceClassifier(nn.Module):
    def __init__(
        self, config, embedder, max_position_embeddings, char_set: set = None, ocr: OCRHead = None, alpha: float = 1
    ):
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
        self.char_set = char_set
        self.positional = PositionalEncoding(config.emb_size, config.dropout, max_position_embeddings)
        self.ctc_criterion = CTCLoss(reduction="sum", zero_infinity=True)
        self.alpha = alpha
        self.num_classes = config.num_classes

    def forward(self, input_batch: dict[str, list | torch.Tensor]) -> dict[str, torch.Tensor]:

        output = self.embedder(input_batch)  # batch_size, seq_len, emb_size
        output["embeddings"] = self.positional(output["embeddings"])
        assert isinstance(input_batch["labels"], torch.Tensor)
        result = self.backbone(
            inputs_embeds=output["embeddings"], labels=input_batch["labels"].to(torch.int64)
        )  # batch_size, num_classes

        if self.ocr:
            assert isinstance(self.embedder, VTREmbedder)
            result["ce_loss"] = result["loss"]
            assert isinstance(input_batch["texts"], list)
            result["ctc_loss"] = compute_ctc_loss(
                self.ctc_criterion, self.ocr, output["ocr_embeddings"], input_batch["texts"], self.char_set
            )
            result["loss"] = result["ce_loss"] + self.alpha * result["ctc_loss"]

        return result
