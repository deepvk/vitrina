import argparse
import json
import os.path

import numpy as np
from razdel import tokenize

from scripts.clusterization import clusterization
from src.utils.common import load_json, clean_text, save_json


CLUSTERIZATION_FILE = "resources/letter_replacement/clusterization.json"
REPLACEMENT_FILE_TEMPLATE = "resources/letter_replacement/letters%s.json"
OBSCENE_AUGMENTED_FILE = "resources/obscene_augmented.txt"


def add_noise(
    text: str,
    similar_symbols: dict[str, list[str]],
    replacement_proba: float,
    add_space_proba: float = 0,
) -> str:
    result = ""

    for ch in text:
        replace = np.random.binomial(1, replacement_proba)
        lower_ch = ch.lower()
        if replace and lower_ch in similar_symbols and len(similar_symbols[lower_ch]) != 0:
            symbols = list(similar_symbols[lower_ch])
            result += symbols[np.random.randint(len(symbols))]
        else:
            result += ch
        add_space = np.random.binomial(1, add_space_proba)
        if add_space:
            result += " "
    return result


def get_letter_replacements(level: int, add_points_and_stars: bool = False):
    letter_replacement = {}

    if level > 0:
        replacement_file = REPLACEMENT_FILE_TEMPLATE.format(min(level, 3))
        with open(replacement_file, "r") as json_file:
            letter_replacement = json.load(json_file)
        if add_points_and_stars:
            for replacements in letter_replacement.values():
                replacements.append(".")
                replacements.append("*")

    if level == 4:
        if not os.path.exists(CLUSTERIZATION_FILE):

            clusterization()

        with open(CLUSTERIZATION_FILE, "r") as json_file:
            clusters = json.load(json_file)
            for key in letter_replacement:
                letter_replacement[key].extend(
                    list(set(clusters[key]).union(set(clusters[key.upper()])) - set(letter_replacement[key]))
                )
    return letter_replacement


def generate(
    data: str,
    level_toxic: int = 4,
    level_non_toxic: int = 2,
    p_toxic_word: float = 0.5,
    p_toxic_symbol: float = 0.5,
    p_space: float = 0.01,
    p_non_toxic_symbol: float = 0.1,
    sl: bool = False,
    save_to: str = "resources/data/noisy_dataset.jsonl",
):
    """
    Generates a noisy dataset based on the original one.

    The noise level can be adjusted with parameters. level-toxic and level-non-toxic define sets from which substitutions are taken for characters with a certain probability.
    The values of these parameters can be {1, 2, 3, 4}. Each next level includes replacements from the previous one.

    1 ‒ replacement with visually similar numbers

    2 ‒ replacement with visually similar characters from the standard keyboard layout or from the Latin alphabet

    3 ‒ replacement with visually similar character sequences

    4 ‒ replacement with characters from the same cluster (see scripts/clusterization.py)

    :param data: path to source dataset
    :param level_toxic: noise level of toxic words (default: 4)
    :param level_non_toxic: noise level of non-toxic words (default: 2)
    :param p_toxic_word: probability of making changes to the toxic word (default: 0.5)
    :param p_toxic_symbol: probability of changing a character in a toxic word (default: 0.5)
    :param p_space: probability of adding a space after a character (default: 0.01)
    :param p_non_toxic_symbol: probability of changing a character in a non-toxic word (default: 0.1)
    :param sl: if the flag is set, then a dataset will be generated for the sequence labeling task (marking offensive words)
    :param save_to: path to the file where the noisy dataset will be saved (default: resources/data/noisy_dataset.jsonl)
    :return:
    """
    labeled_texts = load_json(data)

    toxic_letter_replacement = get_letter_replacements(level_toxic, add_points_and_stars=True)
    non_toxic_letter_replacement = get_letter_replacements(level_non_toxic, add_points_and_stars=False)

    toxic_dict = set()
    with open(OBSCENE_AUGMENTED_FILE, "r") as dict_file:
        for line in dict_file:
            toxic_dict.add(line.strip())

    for labeled_text in labeled_texts:
        text = labeled_text["text"]

        text = clean_text(text)

        tokens = list(tokenize(text))
        labeled_tokens = []

        for token_ind in range(len(tokens)):
            token_text = tokens[token_ind].text
            token_label = 0
            if token_text in toxic_dict:
                replace_chars = np.random.binomial(1, p_toxic_word)
                if replace_chars:
                    token_text = add_noise(token_text, toxic_letter_replacement, p_toxic_symbol, p_space)
                token_label = 1
            else:
                result = add_noise(token_text, non_toxic_letter_replacement, p_non_toxic_symbol, add_space_proba=0)
                token_text = result

            if sl:
                labeled_tokens.append((token_text, token_label))
            else:
                if token_ind != len(tokens) - 1 and tokens[token_ind].stop != tokens[token_ind + 1].start:
                    token_text += " "
                tokens[token_ind].text = token_text

        labeled_text["text"] = labeled_tokens if sl else "".join(list(map(lambda substr: substr.text, tokens)))

    save_json(labeled_texts, save_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="resources/data/dataset.jsonl")
    parser.add_argument("--level-toxic", type=int, default=4)
    parser.add_argument("--level-non-toxic", type=int, default=2)
    parser.add_argument("--p-toxic-word", type=float, default=0.5)
    parser.add_argument("--p-toxic-symbol", type=float, default=0.5)
    parser.add_argument("--p-space", type=float, default=0.01)
    parser.add_argument("--p-non-toxic-symbol", type=float, default=0.1)

    parser.add_argument("--sl", action="store_true")
    parser.add_argument(
        "--save-to",
        type=str,
        default="resources/data/noisy_dataset.jsonl",
    )

    args = parser.parse_args()
    generate(**vars(args))
