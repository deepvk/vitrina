from transformers import BertConfig, BertForSequenceClassification
import torch
from torch import nn
from loguru import logger
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CTCLoss

from src.models.vtr.ocr import OCRHead
from src.utils.common import PositionalEncoding
from src.models.embedders.vtr import VTREmbedder
from src.utils.common import char2int


def compute_ctc_loss(
    criterion: torch.nn.modules.loss.CTCLoss, ocr: OCRHead, embeddings: torch.Tensor, texts: list, char_set: set
):
    logits = ocr(embeddings)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    input_lengths = torch.LongTensor([log_probs.shape[0]] * log_probs.shape[1])

    chars = list("".join(np.concatenate(texts).flatten()))
    targets = char2int(chars, char_set)

    get_len = np.vectorize(len)
    target_lengths = pad_sequence([torch.from_numpy(get_len(arr)) for arr in texts], batch_first=True, padding_value=0)

    ctc_loss = criterion(log_probs, targets, input_lengths, target_lengths)
    ctc_loss /= len(texts)

    return ctc_loss


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
