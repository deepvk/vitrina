import numpy as np
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

from src.utils.common import clean_text
from src.utils.slicer import VTRSlicer


def collate_batch_common(slices: list[torch.Tensor], labels: list[int]):
    # [batch size; most slices; font size; window size]
    batched_slices = pad_sequence(slices, batch_first=True, padding_value=0.0).float()
    bs, ms, _, _ = batched_slices.shape

    # [batch size; most slices]
    attention_mask = torch.zeros((bs, ms), dtype=torch.float)
    for i, s in enumerate(slices):
        attention_mask[i, : len(s)] = 1

    return {
        "slices": batched_slices,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.float),
    }


class DatasetNLLB(IterableDataset):
    def __init__(
        self,
        char2array: dict,
        probas: dict,
        window_size: int = 32,
        stride: int = 5,
        max_seq_len: int = 512,
        random_seed: int = 42,
        k: float = 0.3, 
    ):
        self.datasets = dict()
        self.slicer = VTRSlicer(char2array=char2array, window_size=window_size, stride=stride)
        self.max_seq_len = max_seq_len
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

        np.random.seed(random_seed)

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
                slices = self.slicer(text)
                slices = slices[: self.max_seq_len]

                yield slices, label

    def collate_function(self, batch: list[tuple[torch.Tensor, int]]) -> dict[str, torch.Tensor]:
        slices, labels = [list(item) for item in zip(*batch)]
        return collate_batch_common(slices, labels)


class FloresDataset(Dataset):
    def __init__(
        self,
        lang2label: dict,
        char2array: dict,
        window_size: int = 32,
        stride: int = 5,
        max_seq_len: int = 512,
        split="dev",
    ):
        assert split in ["dev", "devtest"], "Split for FLORES dataset must be dev or devtest"
        self.lang2label = lang2label
        self.langs = lang2label.keys()
        self.slicer = VTRSlicer(char2array=char2array, window_size=window_size, stride=stride)
        self.dataset = load_dataset("facebook/flores", "all")[split]
        self.data = []
        self.max_seq_len = max_seq_len
        for lang in self.langs:
            column_name = f"sentence_{lang}"
            sentences = self.dataset[column_name]
            current_label = self.lang2label[lang]
            for sentence in sentences:
                text = clean_text(sentence)
                self.data.append({"text": text, "label": current_label})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        # [n slices; font size; window size]
        slices = self.slicer(self.data[index]["text"])
        slices = slices[: self.max_seq_len]
        return slices, self.data[index]["label"]

    def collate_function(self, batch: list[tuple[torch.Tensor, int]]) -> dict[str, torch.Tensor]:
        slices, labels = [list(item) for item in zip(*batch)]
        return collate_batch_common(slices, labels)
