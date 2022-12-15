from typing import Sized, TypeVar, Generic, Dict, List

import torch
from torch.utils.data import Dataset

T = TypeVar("T")


class SizedCollatedDataset(Dataset, Sized, Generic[T]):
    def __init__(self, labeled_texts: List[Dict[str, T]]):
        self.labeled_texts = labeled_texts

    def __len__(self):
        return len(self.labeled_texts)

    def __getitem__(self, index) -> Dict[str, List]:
        return self.labeled_texts[index]

    def collate_function(self, batch: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()