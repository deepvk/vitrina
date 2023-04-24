import json
import math
import os
import random
import re
from typing import Callable

import numpy as np
import pymorphy2
import torch
import ujson
from PIL import ImageFont, Image, ImageDraw
from loguru import logger
from torch import nn
from torch.nn.utils.rnn import pad_sequence

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
    criterion: torch.nn.modules.loss.CTCLoss, ocr: OCRHead, embeddings: torch.Tensor, texts: list, char2int_dict: dict
):
    logits = ocr(embeddings)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    input_lengths = torch.LongTensor([log_probs.shape[0]] * log_probs.shape[1])

    chars = list("".join(np.concatenate(texts).flatten()))
    targets = torch.LongTensor([char2int_dict.get(c, char2int_dict["UNK"]) for c in chars])

    get_len = np.vectorize(len)
    #target_lengths = pad_sequence([torch.from_numpy(get_len(arr)) for arr in texts], batch_first=True, padding_value=0)
    target_lengths = []
    for arr in texts:
        if not len(arr):
            continue
        target_lengths.append(torch.from_numpy(get_len(arr)))
    target_lengths = torch.cat(target_lengths, dim=0)
    ctc_loss = criterion(log_probs, targets, input_lengths, target_lengths)
    ctc_loss /= len(texts)

    return ctc_loss


def masking(num_patches, ratio=0.25):
    masked_patches_idx = set()
    neighbours_idx = set()
    failures = 0
    mask = np.zeros(shape=num_patches, dtype=np.int)

    max_masked_patches = math.floor(num_patches * ratio)
    if max_masked_patches == 0:
        return mask

    while len(masked_patches_idx) < max_masked_patches and failures < 100:
        mask_len = random.randint(1, min(6, max_masked_patches))
        left_patch = random.randint(0, num_patches - mask_len)
        right_patch = left_patch + mask_len

        cur_masked = set(range(left_patch, right_patch))
        cur_neighbours = set(range(left_patch - mask_len, left_patch)).union(range(right_patch, right_patch + mask_len))

        if not cur_masked.intersection(neighbours_idx) and not cur_neighbours.intersection(masked_patches_idx):
            masked_patches_idx.update(cur_masked)
            neighbours_idx.update(cur_neighbours)
            failures = 0
        else:
            failures += 1

    mask[list(masked_patches_idx)] = 1

    return mask


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

        #self.dropout = nn.Dropout(p=dropout)

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
        return x
