import torch.nn as nn
import torch
from transformers import BertModel, BertConfig
import numpy as np
from torch.nn import CTCLoss
import lpips

from src.utils.common import PositionalEncoding, compute_ctc_loss
from src.utils.masking import SpanMaskingGenerator
from src.models.vtr.ocr import OCRHead


class Pretrain(nn.Module):
    def __init__(
            self,
            emb_size: int = 512,
            n_head: int = 8,
            n_layers: int = 4,
            device: str = "cpu",
            ocr: OCRHead = None,
            char2int_dict: dict = None,
            alpha: float = 1,
    ):
        super().__init__()
        config = BertConfig(hidden_size=emb_size, num_attention_heads=n_head)
        self.encoder = BertModel(config)
        self.positional_enc = PositionalEncoding(emb_size)
        self.positional_dec = PositionalEncoding(emb_size)
        self.masking = SpanMaskingGenerator(emb_size)
        self.linear = nn.Linear(emb_size, emb_size)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_head, dim_feedforward=emb_size), num_layers=n_layers
        )
        self.ocr = ocr
        self.ctc_criterion = CTCLoss(reduction="sum", zero_infinity=True)
        self.char2int_dict = char2int_dict
        self.device = device
        self.criterion = lpips.LPIPS(net="vgg").to(device)
        self.emb_size = emb_size
        self.alpha = alpha

    def forward(self, input_batch: dict[str, list | torch.Tensor]):

        batch_size, slice_count, height, width = input_batch["slices"].shape
        slices = input_batch["slices"].view(batch_size, slice_count, height * width)
        pos_slices = self.positional_enc(slices)
        slices_detached = pos_slices.detach()

        grey_slice = torch.full((height, width), 128, dtype=torch.float32, device=self.device).flatten()
        masks = []
        unmasked_slices = []
        unmasked_texts = []
        for i in range(batch_size):
            mask = self.masking(self.emb_size)
            masks.append(mask)
            masked_idx = np.where(mask == 1)[0]
            slices_detached[i][masked_idx] = grey_slice
            unmasked_idx = np.where(mask == 0)[0]
            unmasked_slices.append(slices[i][unmasked_idx])
            if self.ocr:
                unmasked_texts.append(np.array(input_batch["texts"][i])[unmasked_idx[: len(input_batch["texts"][i]) - 1]])
        unmasked_slices = torch.stack(unmasked_slices)

        slices = self.linear(slices_detached)
        encoded_text = self.encoder(inputs_embeds=slices)

        slice_count = unmasked_slices.shape[1]
        unmasked_slices = unmasked_slices.view(batch_size * slice_count, 1, height, width)

        encoded_text = self.positional_dec(encoded_text[0])
        decoded_text = self.decoder(encoded_text)

        masked_slices = []
        masked_originals = []
        for i in range(batch_size):
            masked_idx = np.where(masks[i] == 1)[0]
            masked_slices.append(decoded_text[i][masked_idx])
            masked_originals.append(input_batch["slices"][i][masked_idx])
        masked_originals = torch.stack(masked_originals)
        masked_slices = torch.stack(masked_slices)
        slice_count = masked_slices.shape[1]
        masked_slices = masked_slices.view(batch_size, slice_count, height, width)

        result = {"loss": self.criterion(masked_slices, masked_originals).sum()}
        if self.ocr:
            result["ctc_loss"] = compute_ctc_loss(
                self.ctc_criterion, self.ocr, unmasked_slices, unmasked_texts, self.char2int_dict
            )
            result["lpips_loss"] = result["loss"]
            result["loss"] += self.alpha * result["ctc_loss"]

        return result
