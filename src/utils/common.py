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
    return morph.parse(word)[0].normal_form.replace("??", "??")


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
