import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from src.datasets.common import DatasetSample
from src.utils.common import clean_text
from src.utils.slicer import VTRSlicer, VTRSlicerWithText


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


class VTRDataset(Dataset):
    def __init__(
        self,
        labeled_texts: list[DatasetSample],
        char2array: dict,
        window_size: int = 30,
        stride: int = 5,
        max_seq_len: int = 512,
    ):
        logger.info(f"Initializing VTRDataset with {len(labeled_texts)} samples, use max seq len {max_seq_len}")
        self.labeled_texts = labeled_texts
        self.max_seq_len = max_seq_len

        self.slicer = VTRSlicer(char2array=char2array, window_size=window_size, stride=stride)

    def __len__(self) -> int:
        return len(self.labeled_texts)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        sample = self.labeled_texts[index]
        raw_text = clean_text(sample["text"])

        # [n slices; font size; window size]
        slices = self.slicer(raw_text)
        slices = slices[: self.max_seq_len]
        return slices, sample["label"]

    def collate_function(self, batch: list[tuple[torch.Tensor, int]]) -> dict[str, torch.Tensor]:
        slices, labels = [list(item) for item in zip(*batch)]
        return collate_batch_common(slices, labels)


class VTRDatasetOCR(Dataset):
    def __init__(
        self,
        labeled_texts: list[DatasetSample],
        font: str,
        font_size: int = 15,
        window_size: int = 30,
        stride: int = 5,
        max_seq_len: int = 512,
        ratio: float = 0.7,
    ):
        logger.info(f"Initializing VTRDatasetOCR with {len(labeled_texts)} samples, use max seq len {max_seq_len}")

        self.texts = []
        self.labels = []
        self.char_set: set = set()
        for sample in tqdm(labeled_texts):
            cleaned_text = clean_text(sample["text"])
            if cleaned_text:
                self.texts.append(cleaned_text)
                self.labels.append(sample["label"])
                self.char_set |= set(cleaned_text)
        logger.info(f"Got {len(self.texts)} clean samples out of {len(labeled_texts)}")

        self.max_seq_len = max_seq_len

        self.slicer = VTRSlicerWithText(
            font=font, font_size=font_size, window_size=window_size, stride=stride, ratio=ratio
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index) -> tuple[torch.Tensor, int, list[str]]:
        sample = self.texts[index]

        # [n slices; font size; window size]
        slices, slice_text = self.slicer(sample)
        slices = slices[: self.max_seq_len]
        slice_text = slice_text[: self.max_seq_len]
        return slices, self.labels[index], slice_text

    def collate_function(
        self, batch: list[tuple[torch.Tensor, int, list[str]]]
    ) -> dict[str, torch.Tensor | list[list[str]]]:
        slices, labels, texts = [list(item) for item in zip(*batch)]

        collated_batch = collate_batch_common(slices, labels)
        collated_batch["texts"] = texts

        return collated_batch
