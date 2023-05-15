import json
import math
import os
import random
import re
from typing import Callable
import matplotlib.pyplot as plt
import wandb

import numpy as np
import pymorphy2
import torch
import ujson
from PIL import ImageFont, Image, ImageDraw
from loguru import logger
from torch import nn

from src.models.vtr.ocr import OCRHead

morph = pymorphy2.MorphAnalyzer()

_MENTION_REGEXP = re.compile(r"^\[id\d*|.*\],*\s*")
_HTML_ESCAPE_CHR_REGEXP = re.compile(r"(&quot;)|(&lt;)|(&gt;)|(&amp;)|(&apos;)")
_HTML_CODED_CHR_REGEXP = re.compile(r"(&#\d+;)")
_URL_REGEXP = re.compile(r"https?://(www\.)?[-a-zA-Z0-9@:%._+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_+.~#?&/=]*)")
_BR_TOKEN_REGEXP = re.compile(r"<br>")
_BOM_REGEXP = re.compile(r"\ufeff")
_ZERO_WIDTH_SPACE_REGEXP = re.compile(r"\u200b")


def text2image(text: str, font: str, font_size: int = 15) -> Image:
    image_font = ImageFont.truetype(font, max(font_size - 4, 8))
    if len(text) == 0:
        text = " "

    line_width, _ = image_font.getsize(text)

    image = Image.new("L", (line_width, font_size), color="#FFFFFF")
    draw = ImageDraw.Draw(image)

    draw.text(xy=(0, 0), text=text, fill="#000000", font=image_font)

    return image


def load_json(filename: str) -> list:
    with open(filename, encoding="utf-8") as f:
        result = [ujson.loads(line) for line in f]
    return result


def save_json(data: list, filename: str) -> None:
    with open(filename, "w") as outfile:
        for labeled_text in data:
            json.dump(labeled_text, outfile, ensure_ascii=False)
            outfile.write("\n")


def dict_to_device(
    batch: dict[str, torch.Tensor],
    except_keys: set[str] = None,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device is not specified, using {device}")

    except_keys_set: set[str] = except_keys or set()

    for key, val in batch.items():
        if key in except_keys_set:
            continue
        batch[key] = val.to(device)
    return batch


def clean_text(text: str):

    text = re.sub(_MENTION_REGEXP, "", text)
    text = re.sub(_HTML_ESCAPE_CHR_REGEXP, " ", text)
    text = re.sub(_URL_REGEXP, " ", text)
    text = re.sub(_BR_TOKEN_REGEXP, " ", text)
    text = re.sub(_HTML_CODED_CHR_REGEXP, " ", text)
    text = re.sub(_BOM_REGEXP, " ", text)
    text = re.sub(_ZERO_WIDTH_SPACE_REGEXP, "", text)

    return text


def cosine_decay_scheduler(final_steps: int = 300000, warm_steps: int = 3000) -> Callable[[int], float]:
    def scheduler(i: int):
        if i < warm_steps:
            lr_mult = float(i) / float(max(1, warm_steps))
        else:
            progress = float(i - warm_steps) / float(max(1, final_steps - warm_steps))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))
        return lr_mult

    return scheduler


def lemmatize_word(word: str) -> str:
    return morph.parse(word)[0].normal_form.replace("ё", "е")


def set_deterministic_mode(seed: int):
    _set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ctc_loss(
    criterion: torch.nn.modules.loss.CTCLoss, ocr: OCRHead, embeddings: torch.Tensor, texts: list, char2int: dict
):
    logits = ocr(embeddings)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    input_lengths = torch.LongTensor([log_probs.shape[0]] * log_probs.shape[1])

    chars = list("".join(np.concatenate(texts).flatten()))
    targets = torch.LongTensor([char2int.get(c, char2int["UNK"]) for c in chars])

    get_len = np.vectorize(len)
    # target_lengths = pad_sequence([torch.from_numpy(get_len(arr)) for arr in texts], batch_first=True, padding_value=0)
    target_lengths = []
    for arr in texts:
        target_lengths.append(torch.from_numpy(get_len(arr)))
    target_lengths_concat = torch.cat(target_lengths, dim=0)
    ctc_loss = criterion(log_probs, targets, input_lengths, target_lengths_concat)
    ctc_loss /= len(texts)

    return ctc_loss


def create_noise_mask(batch_size, seq_len, not_padded, noise_density=0.8, span_lens=6, min_unmasked=4):
    # 1. Create random mask
    rand_nums = torch.rand(batch_size, seq_len, device=not_padded.device)
    mask = rand_nums.le(noise_density)

    # 2. Calculate position of each mask in span
    cum_sum = torch.cumsum(mask, dim=-1)
    cum_sum_not_mask = torch.masked_fill(cum_sum, mask, 0)
    mask_position = cum_sum - torch.cummax(cum_sum_not_mask, dim=-1).values

    # 3. Avoid too long spans
    min_values = torch.min(span_lens * torch.ones_like(not_padded), not_padded // 2)
    mask = torch.where(mask_position > min_values, torch.zeros_like(mask), mask)

    # 4. Ensure at least min_unmasked unmasked patches after each block
    not_mask = ~mask
    cum_sum_unmasked = torch.cumsum(not_mask, dim=-1)
    cum_sum_masked = torch.masked_fill(cum_sum_unmasked, not_mask, 0)
    unmasked_position = cum_sum_unmasked - torch.cummax(cum_sum_masked, dim=-1).values

    """
    Get indexes of the first patches of unmasked blocks. 
    Calculate next min_unmasked-1 indexes.
    Mark all retrieved indexes as unmasked.
    """
    first_unmasked_idx = torch.nonzero(unmasked_position == 1, as_tuple=True)
    extra_idx = torch.arange(0, min_unmasked, device=unmasked_position.device).view(1, -1)
    unmasked_spans_idx = first_unmasked_idx[1].unsqueeze(1) + extra_idx
    unmasked_spans_idx = torch.clamp(unmasked_spans_idx, 0, unmasked_position.shape[1] - 1)
    batch_idx = first_unmasked_idx[0].unsqueeze(1).repeat(1, min_unmasked)

    unmasked_position[batch_idx.flatten(), unmasked_spans_idx.flatten()] = 1
    mask[unmasked_position > 0] = False

    return mask


def plot_slices(
    decoded: tuple[torch.Tensor, torch.Tensor],
    orig: tuple[torch.Tensor, torch.Tensor],
    iter_num: int,
    loss: float,
    folder_name: str = None,
):
    fig, axs = plt.subplots(2, 2, figsize=(4, 4))
    axs[0, 0].imshow(decoded[0].squeeze(0).cpu().detach().numpy())
    axs[0, 0].set_title("Decoded image 1")
    axs[0, 1].imshow(orig[0].squeeze(0).cpu().detach().numpy())
    axs[0, 1].set_title("Original image 1")
    axs[1, 0].imshow(decoded[1].squeeze(0).cpu().detach().numpy())
    axs[1, 0].set_title("Decoded image 2")
    axs[1, 1].imshow(orig[1].squeeze(0).cpu().detach().numpy())
    axs[1, 1].set_title("Original image 2")
    fig.suptitle(f"Iteration #{iter_num + 1}")
    plt.figtext(0.5, 0.1, f"Loss = {loss}", ha="center")

    if folder_name:
        file_name = str(iter_num + 1) + ".png"
        plt.savefig(os.path.join(folder_name, file_name))
        wandb.log({"Slice plots": wandb.Image(folder_name + "/" + file_name)})
    plt.show()


class BceLossForTokenClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = outputs.view(-1)
        labels = labels.view(-1).float()
        mask = (labels >= 0).float()
        num_tokens = int(torch.sum(mask))
        loss = self.bce_loss(outputs, labels) * mask
        return torch.sum(loss) / num_tokens


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.register_buffer("positions", torch.arange(max_len, dtype=torch.long).reshape(1, -1))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.positions, torch.Tensor)
        pos_emb = self.pos_emb(self.positions[:, : x.shape[1]])  # [1, seq_len, hidden_size]
        x = x + pos_emb
        return self.dropout(x)
