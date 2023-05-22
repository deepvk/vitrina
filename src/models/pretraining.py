import torch.nn as nn
import torch
import numpy as np
from torch.nn import CTCLoss
import lpips
import datetime
import os

from src.utils.common import PositionalEncoding, compute_ctc_loss, create_noise_mask, plot_slices
from src.models.vtr.ocr import OCRHead

GREY = 128
MAX_COLOUR = 255
AVER_LETTER_WIDTH = 6


class MaskedVisualLM(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        height: int = 16,
        width: int = 32,
        emb_size: int = 512,
        no_verbose: bool = False,
        save_plots: bool = False,
        ocr: OCRHead = None,
        char2int: dict = None,
        alpha: float = 1,
    ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_head, dim_feedforward=emb_size),
            num_layers=n_layers,
        )
        self.positional_enc = PositionalEncoding(emb_size)
        self.positional_dec = PositionalEncoding(emb_size)

        self.emb1 = nn.Linear(height * width, emb_size, bias=False)
        self.letter_count = width // AVER_LETTER_WIDTH
        self.linear = nn.Linear(emb_size, 2 * self.letter_count * height)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_head, dim_feedforward=emb_size),
            num_layers=n_layers,
        )
        self.emb2 = nn.Linear(emb_size, height * width, bias=False)
        self.ocr = ocr
        self.ctc_criterion = CTCLoss(reduction="sum", zero_infinity=True)
        self.char2int = char2int
        self.criterion = lpips.LPIPS(net="vgg", lpips=False)
        # self.criterion = nn.MSELoss()
        self.alpha = alpha
        self.iter = 0
        self.dropout = nn.Dropout(dropout)

        self.verbose = not no_verbose
        if save_plots:
            current_datetime = datetime.datetime.now()
            self.folder_name = "resources/plots/" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(self.folder_name)
        else:
            self.folder_name = None

    def forward(self, input_batch: dict[str, list | torch.Tensor]):

        assert isinstance(input_batch["slices"], torch.Tensor)
        batch_size, slice_count, height, width = input_batch["slices"].shape
        slices = input_batch["slices"].view(batch_size, slice_count, height * width).clone()

        assert isinstance(input_batch["attention_mask"], torch.Tensor)
        not_padded = torch.sum(input_batch["attention_mask"], dim=1).view(-1, 1)
        mask = create_noise_mask(batch_size=batch_size, seq_len=slice_count, not_padded=not_padded)
        mask *= input_batch["attention_mask"] == 1
        slices.masked_fill_(mask.unsqueeze(2), GREY)

        slices /= MAX_COLOUR
        slices = self.emb1(slices)
        slices = self.positional_enc(slices).permute(1, 0, 2)

        encoded_text = self.encoder(slices, src_key_padding_mask=input_batch["attention_mask"]).permute(1, 0, 2)
        encoded_text_pos = self.positional_dec(encoded_text).permute(1, 0, 2)
        encoded_text_pos = self.dropout(encoded_text_pos)

        decoded_text = self.decoder(encoded_text_pos, src_key_padding_mask=input_batch["attention_mask"]).permute(
            1, 0, 2
        )
        decoded_text = self.dropout(decoded_text)

        masked_slices = self.emb2(decoded_text[mask])
        masked_originals = input_batch["slices"][mask]

        seq_len = masked_slices.shape[0]
        masked_slices = masked_slices.view(seq_len, 1, height, width)
        masked_originals = masked_originals.unsqueeze(1)

        result = {"loss": self.criterion(masked_slices, masked_originals).sum()}
        if self.ocr:
            no_mask = ~mask
            no_mask *= input_batch["attention_mask"] == 1
            unmasked_texts = []
            for i in range(batch_size):
                unmasked_idx = np.where(no_mask[i].cpu() == 1)[0]
                unmasked_texts.append(np.array(input_batch["texts"][i])[unmasked_idx])
            unmasked_slices = self.linear(encoded_text)[no_mask]
            seq_len = unmasked_slices.shape[0]
            unmasked_slices = unmasked_slices.view(seq_len, 1, height, 2 * self.letter_count)
            result["ctc_loss"] = compute_ctc_loss(
                self.ctc_criterion, self.ocr, unmasked_slices, unmasked_texts, self.char2int
            )
            result["lpips_loss"] = result["loss"].clone()

            result["loss"] += self.alpha * result["ctc_loss"]

        if self.iter % 100 == 0 and self.verbose:
            plot_slices(
                (masked_slices[0], masked_slices[-1]),
                (masked_originals[0], masked_originals[-1]),
                self.iter,
                result["loss"],
                self.folder_name,
            )
        self.iter += 1

        return result
