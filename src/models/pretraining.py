import torch.nn as nn
import torch
import numpy as np
from torch.nn import CTCLoss
import lpips
import matplotlib.pyplot as plt

from src.utils.common import PositionalEncoding, compute_ctc_loss, create_noise_mask
from src.models.vtr.ocr import OCRHead

GREY = 128


class Pretrain(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        height: int = 16,
        width: int = 32,
        ocr: OCRHead = None,
        char2int: dict = None,
        alpha: float = 1,
    ):
        super().__init__()
        self.emb_size = height * width
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=n_head, dim_feedforward=self.emb_size),
            num_layers=n_layers,
        )
        self.positional_enc = PositionalEncoding(self.emb_size)
        self.positional_dec = PositionalEncoding(self.emb_size)
        # self.masking = SpanMaskingGenerator(emb_size)
        self.linear = nn.Linear(self.emb_size, self.emb_size)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=n_head, dim_feedforward=self.emb_size),
            num_layers=n_layers,
        )
        self.ocr = ocr
        self.ctc_criterion = CTCLoss(reduction="sum", zero_infinity=True)
        self.char2int = char2int
        self.criterion = lpips.LPIPS(net="vgg", lpips=False)
        # self.criterion = nn.MSELoss()
        self.alpha = alpha
        self.iter = 0
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_batch: dict[str, list | torch.Tensor]):

        batch_size, slice_count, height, width = input_batch["slices"].shape
        slices = input_batch["slices"].view(batch_size, slice_count, height * width).clone()

        not_padded = torch.sum(input_batch["attention_mask"], dim=1, dtype=torch.float64).view(-1, 1)
        mask = create_noise_mask(batch_size=batch_size, seq_len=slice_count, not_padded=not_padded)
        mask *= input_batch["attention_mask"]
        slices.masked_fill_(mask.unsqueeze(2), GREY)

        slices = self.positional_enc(slices).permute(1, 0, 2)
        # slices = self.linear(slices).permute(1, 0, 2)
        encoded_text = self.encoder(slices, src_key_padding_mask=input_batch["attention_mask"].to(torch.float32)).permute(1, 0, 2)
        encoded_text = self.dropout(encoded_text)

        encoded_text_pos = self.positional_dec(encoded_text).permute(1, 0, 2)
        decoded_text = self.decoder(encoded_text_pos, src_key_padding_mask=input_batch["attention_mask"].to(torch.float32)).permute(1, 0, 2)
        decoded_text = self.dropout(decoded_text)

        masked_slices = decoded_text[mask]
        masked_originals = input_batch["slices"][mask]

        seq_len = masked_slices.shape[0]
        masked_slices = masked_slices.view(seq_len, 1, height, width)
        masked_originals = masked_originals.unsqueeze(1)

        result = {"loss": self.criterion(masked_slices, masked_originals).sum()}
        if self.ocr:
            no_mask = ~mask
            no_mask *= input_batch["attention_mask"]
            unmasked_slices = encoded_text[no_mask]
            seq_len = unmasked_slices.shape[0]
            unmasked_slices = unmasked_slices.view(seq_len, 1, height, width)
            unmasked_texts = []
            for i in range(batch_size):
                unmasked_idx = np.where(no_mask[i].cpu() == 1)[0]
                unmasked_texts.append(np.array(input_batch["texts"][i])[unmasked_idx])
            seq_len = unmasked_slices.shape[0]
            unmasked_slices = unmasked_slices.view(seq_len, 1, height, width)
            result["ctc_loss"] = compute_ctc_loss(
                self.ctc_criterion, self.ocr, unmasked_slices, unmasked_texts, self.char2int
            )
            result["lpips_loss"] = result["loss"].clone()

            result["loss"] += self.alpha * result["ctc_loss"]

        if self.iter % 100 == 0:
            plt.rcParams["figure.figsize"] = (4, 2)
            decoded = masked_slices[0]
            orig = masked_originals[0]
            plt.subplot(1, 2, 1)
            plt.imshow(decoded.squeeze(0).cpu().detach().numpy())
            plt.title("Decoded image")
            plt.subplot(1, 2, 2)
            plt.imshow(orig.squeeze(0).cpu().detach().numpy())
            plt.title("Original image")
            plt.suptitle(f"Iteration #{self.iter+1}")
            plt.figtext(0.5, 0.1, f"Loss = {result['loss']}", ha="center")
            plt.show()
        self.iter += 1

        return result
