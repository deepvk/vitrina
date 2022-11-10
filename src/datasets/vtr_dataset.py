from collections import defaultdict
from typing import Any, Dict, List, Union
import torch
from torch.utils.data import Dataset

from utils.slicer import VTRSlicer
from torch.nn.utils.rnn import pad_sequence

from utils.utils import clean_text, load_json


class VTRDataset(Dataset):
    def __init__(
        self,
        labeled_texts: List[Dict[str, Union[str, int]]],
        font: str,
        font_size: int = 15,
        window_size: int = 30,
        stride: int = 5,
        max_seq_len: int = 512,
    ):
        self.labeled_texts = labeled_texts
        self.slicer = VTRSlicer(
            font=font, font_size=font_size, window_size=window_size, stride=stride
        )
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.labeled_texts)

    def __getitem__(self, index) -> Dict[str, Any]:
        labeled_text = self.labeled_texts[index]

        slices = self.slicer(clean_text(labeled_text["text"]))
        slices = slices[: self.max_seq_len]
        return {
            "slices": slices,
            "attention_mask": torch.ones(slices.shape[0]),
            "labels": labeled_text["label"],
        }

    @staticmethod
    def collate_function(batch: List[Dict[str, torch.Tensor]]) -> Dict:
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


if __name__ == "__main__":
    labeled_texts = load_json("data/vk_toxic.jsonl")
    dataset = VTRDataset(labeled_texts, "fonts/NotoSans.ttf")
    print(dataset[7])
