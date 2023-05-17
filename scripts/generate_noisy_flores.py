import argparse
import json
import os
import pickle

from src.utils.augmentation import TextAugmentationWrapper, init_augmentations
from src.utils.common import save_json
from src.utils.config import AugmentationConfig
from tqdm import tqdm
from datasets import load_dataset


def add_noise_flores(
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

    save_dir = os.path.join(
        save_dir, f"w{expected_changes_per_word}t{expected_changes_per_text}m{max_augmentations}p{proba_per_text}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    dataset = load_dataset("facebook/flores", "all")
    data = []
    for split in dataset.keys():
        for sample in tqdm(dataset[split]):
            for k, v in sample.items():
                if k.startswith("sentence"):
                    sample[k] = augmentation_wrapper(v)
            data.append(sample)

        save_json(data, os.path.join(save_dir, f"{split}.jsonl"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

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
