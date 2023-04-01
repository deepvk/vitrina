import numpy as np
from datasets import load_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm

from src.utils.common import clean_text
from src.utils.slicer import VTRSlicer


class DatasetNLLB(IterableDataset):
    def __init__(
        self,
        char2array: dict,
        probas: dict,
        window_size: int = 30,
        stride: int = 5,
        max_seq_len: int = 512,
        random_seed: int = 42,
        k: float = 1.0,
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

    def __iter__(self):
        while True:
            random_pair = np.random.choice(self.pairs, p=self.probas)
            print(random_pair)
            print(self.datasets)
            try:
                info = next(self.datasets[random_pair])
            except StopIteration:
                dataset = load_dataset("allenai/nllb", random_pair, split="train", streaming=True)
                self.datasets[random_pair] = iter(dataset)
                continue

            for elem in info["translation"].items():
                text = clean_text(elem[1])
                label = self.lang2label[elem[0]]
                slices = self.slicer(text)
                slices = slices[: self.max_seq_len]

                yield slices, label
