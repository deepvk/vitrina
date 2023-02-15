import argparse
import os
import pickle

import ftfy
import numpy as np
from fontTools.ttLib import TTFont
from loguru import logger
from tqdm import tqdm

from src.utils.common import text2image

FILL_VALUE_UNKNOWN = 255


def text2numpy(text: str, font: str, font_size: int = 16):
    image = text2image(text, font=font, font_size=font_size)
    image_bytes = np.array(image)
    image_bytes = 255 - image_bytes
    return image_bytes


# Check whether font supports symbol
def has_glyph(font: TTFont, glyph: str) -> bool:
    if len(glyph) != 1:
        raise ValueError(f"Glyph '{glyph}' has incorrect length: {len(glyph)}")
    for table in font["cmap"].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False


# Find correct font that supports symbol or returns None
def pick_font(symbol: str, fonts: list) -> str | None:
    for font in fonts:
        if has_glyph(TTFont(font), symbol):
            return font
    return None


def build_char2array(
    save_to: str,
    fonts_directory: str,
    chars_path: str,
    font_size: int = 16,
    width_unknown: int = None,
    ignore_invalid_encoding: bool = False,
):
    fonts = [
        os.path.join(fonts_directory, font) for font in os.listdir(fonts_directory) if font.endswith(("ttf", "otf"))
    ]

    with open(chars_path, "rb") as f:
        all_chars = pickle.load(f)

    char2array = dict()
    not_supported = []
    for char in tqdm(all_chars):
        current_font = pick_font(char, fonts)

        if current_font:
            char2array[char] = text2numpy(char, font=current_font, font_size=font_size)
        else:
            not_supported.append(char)

    if not ignore_invalid_encoding:
        for char in not_supported:
            fixed_char, explanation_of_error = ftfy.fix_and_explain(char)
            # if char was fixed and there is explanation of error (char encoding is broken)
            if fixed_char != char and explanation_of_error:
                current_font = pick_font(char, fonts)
                if current_font:
                    char2array[char] = text2numpy(fixed_char, font=current_font, font_size=font_size)
                    not_supported.remove(char)

    if not width_unknown:
        width_unknown = int(np.median(np.array([v.shape[1] for k, v in char2array.items()])))
    char2array["UNK"] = np.full(shape=(font_size, width_unknown), fill_value=FILL_VALUE_UNKNOWN, dtype=np.int_)

    pickle.dump(char2array, open(save_to, "wb"))

    logger.info(
        f"Number of symbols in char2array: {len(char2array)}, len of not_supported symbols: {len(not_supported)}"
    )
    logger.info(f"Char2array covers {round(len(char2array)/len(all_chars)*100,2)}% of all symbols")

    return char2array, not_supported


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-to", type=str, default="resources/char2array.pkl")
    parser.add_argument("--fonts-directory", type=str, default="resources/fonts")
    parser.add_argument("--chars-path", type=str, default="resources/data/chars.txt")
    parser.add_argument("--ignore-invalid-encoding", action="store_true", help="Ignore symbols with invalid encoding")
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--width-unknown", type=int, default=None)

    args = parser.parse_args()
    build_char2array(**vars(args))
