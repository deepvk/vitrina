import torch
from typing import TypedDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, Dataset
from transformers import PreTrainedTokenizer

from src.datasets.translation_datasets import NLLBDataset, FloresDataset
from src.utils.augmentation import TextAugmentationWrapper, AugmentationWord
from src.utils.slicer import VTRSlicer


class DatasetSample(TypedDict):
    """Type definition for a sample in a sequence classification dataset.
    Example:
        {"text": "скотина! что сказать", "label": 1}
    """

    text: str
    label: int


class SLDatasetSample(TypedDict):
    """Type definition for a sample in a sequence labeling dataset.
    Example:
        {"text": [("cкотина", 0), ("!", 0), ("что", 0), ("сказать", 0)], "label": 1}
    """

    text: list[tuple[str, int]]
    label: int


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


class AugmentationDataset(IterableDataset):
    def __init__(
        self,
        dataset: NLLBDataset,
        augmentations: list[tuple[AugmentationWord, float]],
        proba_per_text: float,
        expected_changes_per_text: int,
        max_augmentations: int,
    ):
        self.dataset = dataset
        self.augmentation = TextAugmentationWrapper(
            augmentations=augmentations,
            proba_per_text=proba_per_text,
            expected_changes_per_text=expected_changes_per_text,
            max_augmentations=max_augmentations,
        )

    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            try:
                text, label = next(iterator)
            except StopIteration:
                return
            else:
                noisy_text = self.augmentation(text)
                yield noisy_text, label

    def get_num_classes(self):
        return self.dataset.get_num_classes()


class Tokenized:
    def __init__(
        self,
        dataset: NLLBDataset | AugmentationDataset | FloresDataset,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

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

    def get_num_classes(self):
        return self.dataset.get_num_classes()


class TokenizedIterableDataset(Tokenized, IterableDataset):
    def __iter__(self):
        yield from self.dataset.__iter__()


class TokenizedDataset(Tokenized, Dataset):
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)


class Slices:
    def __init__(
        self,
        dataset: NLLBDataset | AugmentationDataset | FloresDataset,
        char2array: dict,
        window_size: int = 32,
        stride: int = 5,
        max_seq_len: int = 512,
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.slicer = VTRSlicer(char2array=char2array, window_size=window_size, stride=stride)

    def get_num_classes(self):
        return self.dataset.get_num_classes()

    def collate_function(self, batch: list[tuple[torch.Tensor, int]]) -> dict[str, torch.Tensor]:
        slices, labels = [list(item) for item in zip(*batch)]
        return collate_batch_common(slices, labels)


class SlicesIterableDataset(Slices, IterableDataset):
    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            try:
                text, label = next(iterator)
            except StopIteration:
                return
            else:
                slices = self.slicer(text)
                slices = slices[: self.max_seq_len]
                yield slices, label


class SlicesDataset(Slices, Dataset):
    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        text, label = self.dataset.__getitem__(index)

        slices = self.slicer(text)
        slices = slices[: self.max_seq_len]
        return slices, label
