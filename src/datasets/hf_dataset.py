import numpy as np
from datasets import get_dataset_config_names, load_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm

from src.utils.common import clean_text
from src.utils.slicer import VTRSlicer


class DatasetNLLB(IterableDataset):
    def __init__(self, char2array: dict, window_size: int = 30, stride: int = 5, max_seq_len: int = 512):
        self.datasets = dict()
        self.slicer = VTRSlicer(char2array=char2array, window_size=window_size, stride=stride)
        self.max_seq_len = max_seq_len
        self.pairs = get_dataset_config_names("allenai/nllb")
        self.lang2label = {}
        self.label2lang = {}
        label = 0
        for pair in tqdm(self.pairs):
            for lang in pair.split("-"):
                if lang not in self.lang2label:
                    self.lang2label[lang] = label
                    self.label2lang[label] = lang
                    label += 1
            dataset = load_dataset("allenai/nllb", pair, split="train", streaming=True)
            self.datasets[pair] = iter(dataset)

    def __iter__(self):
        while True:
            random_pair = np.random.choice(self.pairs)
            try:
                info = next(self.datasets[random_pair])
            except StopIteration:
                self.pairs.remove(random_pair)
                continue

            src, trg = [elem for elem in info["translation"].items()]

            for elem in (src, trg):
                text = clean_text(elem[1])
                label = self.lang2label[elem[0]]
                slices = self.slicer(text)
                slices = slices[: self.max_seq_len]

                yield slices, label
