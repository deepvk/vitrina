import argparse
import json
import os.path
from typing import List, Dict

import numpy as np
from razdel import tokenize

from clusterization import clusterization
from src.utils.utils import load_json, clean_text, save_json


def add_noise(
    text: str,
    similar_symbols: Dict[str, List[str]],
    replacement_proba: float,
    add_space_proba: float = 0,
) -> str:
    result = ""

    for ch in text:
        replace = np.random.binomial(1, replacement_proba)
        lower_ch = ch.lower()
        if (
            replace
            and lower_ch in similar_symbols
            and len(similar_symbols[lower_ch]) != 0
        ):
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
        replacement_file = f"resources/letter_replacement/letters{min(3, level)}.json"
        with open(replacement_file, "r") as json_file:
            letter_replacement = json.load(json_file)
        if add_points_and_stars:
            for replacements in letter_replacement.values():
                replacements.append(".")
                replacements.append("*")

    if level == 4:
        clusterization_file = "resources/letter_replacement/clusterization.json"
        if not os.path.exists(clusterization_file):
            clusterization()

        with open(clusterization_file, "r") as json_file:
            clusters = json.load(json_file)
            for key in letter_replacement:
                letter_replacement[key].extend(
                    list(
                        set(clusters[key]).union(set(clusters[key.upper()]))
                        - set(letter_replacement[key])
                    )
                )
    return letter_replacement


def generate(
    data: str,
    toxic_level: int,
    non_toxic_level: int,
    change_toxic_word_proba: float,
    repl_toxic_symbol_proba: float,
    add_space_after_toxic_sym_proba: float,
    repl_nontoxic_sym_proba: float,
    sl: bool,
    output: str,
):
    labeled_texts = load_json(data)

    toxic_letter_replacement = get_letter_replacements(
        toxic_level, add_points_and_stars=True
    )
    non_toxic_letter_replacement = get_letter_replacements(
        non_toxic_level, add_points_and_stars=False
    )

    toxic_dict = set()
    with open("resources/obscene_augmented.txt", "r") as dict_file:
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
                replace_chars = np.random.binomial(1, change_toxic_word_proba)
                if replace_chars:
                    token_text = add_noise(
                        token_text,
                        toxic_letter_replacement,
                        repl_toxic_symbol_proba,
                        add_space_after_toxic_sym_proba,
                    )
                token_label = 1
            else:
                result = add_noise(
                    token_text,
                    non_toxic_letter_replacement,
                    repl_nontoxic_sym_proba,
                    add_space_proba=0,
                )
                token_text = result

            if sl:
                labeled_tokens.append((token_text, token_label))
            else:
                if (
                    token_ind != len(tokens) - 1
                    and tokens[token_ind].stop != tokens[token_ind + 1].start
                ):
                    token_text += " "
                tokens[token_ind].text = token_text

        labeled_text["text"] = (
            labeled_tokens
            if sl
            else "".join(list(map(lambda substr: substr.text, tokens)))
        )

    save_json(labeled_texts, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="resources/data/insults.jsonl")
    parser.add_argument("--toxic_level", type=int, default=4)
    parser.add_argument("--non-toxic-level", type=int, default=2)
    parser.add_argument("--change-toxic-word-proba", type=float, default=0.5)
    parser.add_argument("--repl-toxic-symbol-proba", type=float, default=0.5)
    parser.add_argument("--add-space-after-toxic-sym-proba", type=float, default=0.01)
    parser.add_argument("--repl-nontoxic-sym-proba", type=float, default=0.1)

    parser.add_argument("--sl", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="resources/data/noisy_insults.jsonl",
    )

    args = parser.parse_args()
    generate(**vars(args))
