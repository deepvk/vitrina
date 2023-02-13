import argparse
import pickle
from os import listdir
import numpy as np
from src.utils.common import text2image
from fontTools.ttLib import TTFont
from loguru import logger
from tqdm import tqdm
import ftfy


def text2numpy(text: str, font: str, font_size: int = 16):
    image = text2image(text, font=font, font_size=font_size)
    image_bytes = np.array(image)
    image_bytes = 255 - image_bytes
    return image_bytes


# Check whether font supports symbol
def has_glyph(font: TTFont, glyph: str) -> bool:
    assert len(glyph) == 1, glyph
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
    fill_value_unknown: int = 255,
    width_unknown: int = None,
    ignore_invalid_encoding: bool = False,
):
    assert fill_value_unknown in range(256), "Value of filling must be int number in [0,255]"
    fonts = [f"{fonts_directory}/{font}" for font in listdir(fonts_directory) if font.endswith(("ttf", "otf"))]

    with open(chars_path, "rb") as f:
        all_chars = pickle.load(f)

    char2array = dict()
    not_supported = []
    problem_chars = []
    for char in tqdm(all_chars):
        current_font = pick_font(char, fonts)

        if current_font:
            try:
                char2array[char] = text2numpy(char, font=current_font, font_size=font_size)
            except:
                problem_chars.append(char)
        else:
            not_supported.append(char)

    if not ignore_invalid_encoding:
        for char in not_supported:
            fixed_char, explanation_of_error = ftfy.fix_and_explain(char)
            # if char was fixed and there is explanation of error (char encoding is broken)
            if fixed_char != char and explanation_of_error:
                try:
                    current_font = pick_font(char, fonts)
                    if current_font:
                        char2array[char] = text2numpy(fixed_char, font=current_font, font_size=font_size)
                except:
                    problem_chars.append(char)
                else:
                    not_supported.remove(char)

    not_supported += problem_chars
    if not width_unknown:
        width_unknown = int(np.median(np.array([v.shape[1] for k, v in char2array.items()])))
    char2array["UNK"] = np.full(shape=(font_size, width_unknown), fill_value=fill_value_unknown, dtype=np.int_)

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
    parser.add_argument("--chars-path", type=str, default="resources/data/chars.pkl")
    parser.add_argument("--ignore-invalid-encoding", type=bool, default=False)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--width-unknown", type=int, default=None)
    parser.add_argument("--fill-value-unknown", type=int, default=255)

    args = parser.parse_args()
    build_char2array(save_to=args.save_to, fonts_directory=args.fonts_directory, chars_path=args.chars_path)
