from argparse import ArgumentParser

import torch

from datasets.bert_dataset import BERTDataset
from torch.utils.data import random_split

from utils.utils import load_json, save_json


def main(data_path: str, test_size: float, random_seed: int):
    data = load_json(data_path)

    dataset = BERTDataset(data, "berts/toxic-bert")

    dataset_size = len(dataset)
    test_dataset_size = int(test_size * dataset_size)
    train_dataset_size = dataset_size - test_dataset_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_dataset_size, test_dataset_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    train_labeled_texts = []
    for lt in train_dataset:
        train_labeled_texts.append(lt)

    save_json(train_labeled_texts, "data/train_tokenizer_rs21.jsonl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="data/vk_toxic.jsonl")
    parser.add_argument("--random-seed", type=str, default=21)
    parser.add_argument("--test-size", type=float, default=0.1)
    args=parser.parse_args()
    main(data_path=args.data, test_size=args.test_size, random_seed=args.random_seed)