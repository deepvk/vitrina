import argparse
from pathlib import Path

import torch
from tokenizers.implementations import BertWordPieceTokenizer
from torch.utils.data import Dataset, random_split

from utils.utils import load_json


class SimpleTextDataset(Dataset):
    def __init__(self, data_path):
        self.data = load_json(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def train_tokenizer(data_path: str, test_size: float, val_size: float, random_state: int, save_to: str):
    bert_tokenizer = BertWordPieceTokenizer(
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        clean_text=False,
        handle_chinese_chars=False,
        lowercase=False,
        wordpieces_prefix="##"
    )

    dataset = SimpleTextDataset(data_path)

    dataset_size = len(dataset)
    test_dataset_size = int(test_size * dataset_size)
    train_dataset_size = dataset_size - test_dataset_size
    train_dataset, _ = random_split(
        dataset,
        [train_dataset_size, test_dataset_size],
        generator=torch.Generator().manual_seed(random_state),
    )

    train_dataset_size = len(train_dataset)
    val_dataset_size = int((val_size / (1 - test_size)) * train_dataset_size)
    train_dataset_size = train_dataset_size - val_dataset_size
    train_dataset, _ = random_split(
        train_dataset,
        [train_dataset_size, val_dataset_size],
        generator=torch.Generator().manual_seed(random_state)
    )

    texts = list(map(lambda x: x["text"], train_dataset))
    print(texts[0])

    bert_tokenizer.train_from_iterator(texts, vocab_size=30_000,
                                       wordpieces_prefix="##",
                                       special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    output_dir = Path(save_to)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    bert_tokenizer.save_model(save_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=f"resources/data/dataset.jsonl")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=float, default=21)
    parser.add_argument("--save-to", type=str, default=f"tokenizer")

    args = parser.parse_args()
    train_tokenizer(
        data_path=args.data,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        save_to=args.save_to
    )
