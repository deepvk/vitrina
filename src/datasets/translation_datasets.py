import numpy as np
from datasets import load_dataset
from src.utils.common import clean_text
from multiprocessing import Pool
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm
import os


def load_dataset_fn(pair_name):
    return pair_name, load_dataset("allenai/nllb", pair_name, split="train", streaming=True)


class NLLBDataset(IterableDataset):
    def __init__(
        self,
        probas: dict,
        k: float = 0.3,
    ):
        self.datasets = dict()
        self.pairs = list(probas.keys())
        self.probas = []
        self.lang2label: dict = {}
        self.label2lang: dict = {}
        label = 0
        for pair in self.pairs:
            for lang in pair.split("-"):
                if lang not in self.lang2label:
                    self.lang2label[lang] = label
                    self.label2lang[label] = lang
                    label += 1

            self.probas.append(probas[pair] ** k)

        dataset_pairs = [load_dataset_fn(self.pairs[0])]
        with Pool() as pool:
            results = pool.imap(load_dataset_fn, self.pairs[1:], chunksize=1)
            for result in tqdm(results, total=len(self.pairs) - 1):
                dataset_pairs.append(result)
        self.datasets = {pair: iter(dataset) for pair, dataset in dataset_pairs}

        sum_probas = sum(self.probas)
        self.probas = [prob / sum_probas for prob in self.probas]

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


class FloresDataset(Dataset):
    def __init__(self, lang2label: dict, split="dev", dataset_dir: str = None):
        assert split in ["dev", "devtest"], "Split for FLORES dataset must be dev or devtest"
        self.lang2label = lang2label
        self.langs = lang2label.keys()

        if dataset_dir:
            self.dataset = load_dataset("json", data_files=os.path.join(dataset_dir, f"{split}.jsonl"))["train"]
        else:
            self.dataset = load_dataset("facebook/flores", "all")[split]

        self.data = []
        for lang in self.langs:
            column_name = f"sentence_{lang}"
            sentences = self.dataset[column_name]
            current_label = self.lang2label[lang]
            for sentence in sentences:
                text = clean_text(sentence)
                self.data.append({"text": text, "label": current_label})

    def get_data(self):
        return self.data

    def __getitem__(self, index) -> tuple[str, int]:
        return self.data[index]["text"], self.data[index]["label"]

    def __len__(self) -> int:
        return len(self.data)
