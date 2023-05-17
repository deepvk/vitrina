import numpy as np
import torch
from datasets import load_dataset
from src.utils.common import clean_text
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

from transformers import NllbTokenizer


class NLLBDataset(IterableDataset):
    def __init__(
        self,
        probas: dict,
        k: float = 0.3,
        tokenizer: NllbTokenizer = None,
        max_seq_len: int = None,
    ):
        self.datasets = dict()
        self.pairs = list(probas.keys())
        self.probas = []
        self.lang2label: dict = {}
        self.label2lang: dict = {}
        label = 0
        for pair in tqdm(self.pairs):
            for lang in pair.split("-"):
                if lang not in self.lang2label:
                    self.lang2label[lang] = label
                    self.label2lang[label] = lang
                    label += 1
            dataset = load_dataset("allenai/nllb", pair, split="train", streaming=True)
            self.datasets[pair] = iter(dataset)
            self.probas.append(probas[pair] ** k)

        sum_probas = sum(self.probas)
        self.probas = [prob / sum_probas for prob in self.probas]

        if tokenizer:
            self.tokenizer = tokenizer
        if max_seq_len:
            self.max_seq_len = max_seq_len

    def get_num_classes(self):
        return len(self.lang2label)

    def get_lang2label(self):
        return self.lang2label

    def __iter__(self):
        while True:
            random_pair = np.random.choice(self.pairs, p=self.probas)
            try:
                info = next(self.datasets[random_pair])
            except StopIteration:
                dataset = load_dataset("allenai/nllb", random_pair, split="train", streaming=True)
                self.datasets[random_pair] = iter(dataset)

            for elem in info["translation"].items():
                text = clean_text(elem[1])
                if len(text) == 0:
                    continue
                label = self.lang2label[elem[0]]
                yield text, label

    def collate_function(self, batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        tokenized_batch = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokenized_batch["labels"] = torch.tensor(labels, dtype=torch.int64)

        return tokenized_batch


class FloresDataset(Dataset):
    def __init__(
        self,
        lang2label: dict,
        split="dev",
    ):
        assert split in ["dev", "devtest"], "Split for FLORES dataset must be dev or devtest"
        self.lang2label = lang2label
        self.langs = lang2label.keys()
        self.dataset = load_dataset("facebook/flores", "all")[split]
        self.data = []
        for lang in self.langs:
            column_name = f"sentence_{lang}"
            sentences = self.dataset[column_name]
            current_label = self.lang2label[lang]
            for sentence in sentences:
                text = clean_text(sentence)
                self.data.append({"text": text, "label": current_label})

    def get_dataset(self):
        return self.data

    def __getitem__(self, index) -> tuple[str, int]:
        return self.data[index]["text"], self.data[index]["label"]

    def __len__(self) -> int:
        return len(self.data)
