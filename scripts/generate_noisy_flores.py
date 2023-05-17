import argparse
import json
import os
import pickle

from src.datasets.translation_datasets import FloresDataset
from src.utils.augmentation import TextAugmentationWrapper, init_augmentations
from src.utils.common import save_json
from src.utils.config import AugmentationConfig
from tqdm import tqdm


def add_noise_flores(
    probas_path: str,
    leet,
    clusters,
    expected_changes_per_word,
    proba_per_text,
    expected_changes_per_text,
    max_augmentations,
    save_dir,
    split="dev",
):
    assert split in ["dev", "devtest"], "Split for FLORES dataset must be dev or devtest"

    with open(probas_path, "rb") as f:
        probas = pickle.load(f)

    pairs = list(probas.keys())
    lang2label: dict = {}
    label2lang: dict = {}
    label = 0
    for pair in pairs:
        for lang in pair.split("-"):
            if lang not in lang2label:
                lang2label[lang] = label
                label2lang[label] = lang
                label += 1

    flores = FloresDataset(lang2label, split)
    flores_data = flores.get_dataset()

    with open(leet) as json_file:
        leet_symbols = json.load(json_file)

    with open(clusters, "rb") as f:
        cluster_symbols = pickle.load(f)

    augmentations = init_augmentations(
        expected_changes_per_word=expected_changes_per_word, cluster_symbols=cluster_symbols, leet_symbols=leet_symbols
    )

    augmentation_wrapper = TextAugmentationWrapper(
        augmentations=augmentations,
        proba_per_text=proba_per_text,
        expected_changes_per_text=expected_changes_per_text,
        max_augmentations=max_augmentations,
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    noisy_flores = []
    for elem in tqdm(flores_data):
        text = elem["text"]
        noisy_text = augmentation_wrapper(text)
        noisy_flores.append({"text": noisy_text, "label": elem["label"]})

    save_json(
        noisy_flores,
        os.path.join(
            save_dir,
            f"flores_{split}_w{expected_changes_per_word}t{expected_changes_per_text}m{max_augmentations}p{proba_per_text}.jsonl",
        ),
    )
    return noisy_flores


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--probas-path",
        type=str,
        default="resources/nllb/probas_nllb.pkl",
        help="Path to probabilities of language pairs [for lang detect task].",
    )

    arg_parser.add_argument(
        "--save-dir",
        type=str,
        default="resources/data",
        help="Directory for saving noisy dataset",
    )

    arg_parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Split for dataset (only dev or devtest)",
    )
    arg_parser = AugmentationConfig.add_to_arg_parser(arg_parser)
    args = arg_parser.parse_args()
    add_noise_flores(**vars(args))
