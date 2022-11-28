from collections import defaultdict
from typing import Any, Dict, List, Union
import torch
from torch.utils.data import Dataset

from datasets.sized_collated_dataset import SizedCollatedDataset
from utils.slicer import VTRSlicer
from torch.nn.utils.rnn import pad_sequence

from utils.utils import clean_text, load_json


class VTRDataset(SizedCollatedDataset[Union[str, int]]):
    def __init__(
        self,
        labeled_texts: List[Dict[str, Union[str, int]]],
        font: str,
        font_size: int = 15,
        window_size: int = 30,
        stride: int = 5,
        max_seq_len: int = 512,
    ):
        super().__init__(labeled_texts)
        self.slicer = VTRSlicer(
            font=font, font_size=font_size, window_size=window_size, stride=stride
        )
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.labeled_texts)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        labeled_text = self.labeled_texts[index]

        slices = self.slicer(clean_text(labeled_text["text"]))
        slices = slices[: self.max_seq_len]
        return {
            "slices": slices,
            "attention_mask": torch.ones(slices.shape[0]),
            "labels": labeled_text["label"],
        }

    def collate_function(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        key2values = defaultdict(list)
        for example in batch:
            for key, val in example.items():
                key2values[key].append(val)

        return {
            "slices": pad_sequence(key2values["slices"], batch_first=True),
            "attention_mask": pad_sequence(
                key2values["attention_mask"], batch_first=True
            ),
            "labels": torch.tensor(key2values["labels"], dtype=torch.float),
        }
