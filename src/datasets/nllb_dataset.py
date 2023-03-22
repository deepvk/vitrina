import numpy as np
from datasets import get_dataset_config_names, load_dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm

from src.utils.common import clean_text
from src.utils.slicer import VTRSlicer


class DatasetNLLB(IterableDataset):
    def __init__(
        self,
        pairs,
        char2array: dict,
        probas: dict,
        window_size: int = 30,
        stride: int = 5,
        max_seq_len: int = 512,
        random_seed: int = 42,
    ):
        self.datasets = dict()
        self.slicer = VTRSlicer(char2array=char2array, window_size=window_size, stride=stride)
        self.max_seq_len = max_seq_len
        self.pairs = []
        self.probas = []
        self.lang2label: dict = {}
        self.label2lang: dict = {}
        label = 0
        for pair in tqdm(pairs[:2]):
            for lang in pair:
                if lang not in self.lang2label:
                    self.lang2label[lang] = label
                    self.label2lang[label] = lang
                    label += 1
            pair_name = f"{pair[0]}-{pair[1]}"
            dataset = load_dataset("allenai/nllb", pair_name, split="train", streaming=True)
            self.datasets[pair_name] = iter(dataset)
            self.pairs.append(pair_name)
            self.probas.append(probas[pair_name])

        np.random.seed(random_seed)

    def __iter__(self):
        while True:
            random_pair = np.random.choice(self.pairs)
            try:
                info = next(self.datasets[random_pair])
            except StopIteration:
                self.probas.pop(self.pairs.index(random_pair))
                self.pairs.remove(random_pair)
                continue

            for elem in info["translation"].items():
                text = clean_text(elem[1])
                label = self.lang2label[elem[0]]
                slices = self.slicer(text)
                slices = slices[: self.max_seq_len]

                yield slices, label
