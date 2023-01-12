from typing import Any

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.datasets.common import DatasetSample
from src.utils.common import clean_text
from src.utils.slicer import VTRSlicer


class VTRDataset(Dataset):
    def __init__(
        self,
        labeled_texts: list[DatasetSample],
        font: str,
        font_size: int = 15,
        window_size: int = 30,
        stride: int = 5,
        max_seq_len: int = 512,
        ratio: float = 0.7,
        ocr_flag: bool = True,
    ):
        logger.info(f"Initializing VTRDataset with {len(labeled_texts)} samples, use max seq len {max_seq_len}")
        self.labeled_texts = labeled_texts
        self.max_seq_len = max_seq_len
        self.ocr_flag = ocr_flag

        self.slicer = VTRSlicer(font=font, font_size=font_size, window_size=window_size, stride=stride, ratio=ratio)

    def __len__(self) -> int:
        return len(self.labeled_texts)

    def __getitem__(self, index) -> tuple[Any, Any, str | int]:
        sample = self.labeled_texts[index]
        raw_text = clean_text(sample["text"])

        # [n slices; font size; window size]
        if self.ocr_flag:
            slices, slice_text = self.slicer(raw_text)
        slices = slices[: self.max_seq_len]
        slice_text = slice_text[: self.max_seq_len]

        return slices, sample["label"], slice_text

    def collate_function(self, batch: list[tuple[torch.Tensor, int]]) -> dict[str, torch.Tensor]:
        slices = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        texts = [item[2] for item in batch]

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
            "texts": texts
        }
