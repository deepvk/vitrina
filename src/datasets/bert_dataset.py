from collections import defaultdict
from typing import List, Dict, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

from src.datasets.sized_collated_dataset import SizedCollatedDataset
from src.utils.utils import clean_text


class BERTDataset(SizedCollatedDataset[Union[str, int]]):
    PADDED_VECTORS = ["input_ids", "attention_mask", "token_type_ids"]

    def __init__(
        self,
        labeled_texts: List[Dict[str, Union[str, int]]],
        tokenizer: str,
        max_seq_len: int = 512,
    ):
        super().__init__(labeled_texts)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len

    def __getitem__(self, index) -> Dict[str, List]:
        labeled_text = self.labeled_texts[index]

        encoded_dict = self.tokenizer(
            clean_text(labeled_text["text"]),
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
        )

        encoded_dict["labels"] = labeled_text["label"]
        return encoded_dict

    def collate_function(self, batch: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        key2values = defaultdict(list)
        for item in batch:
            for key, val in item.items():
                key2values[key].append(torch.tensor(val))

        padded_batch = {}
        for key in BERTDataset.PADDED_VECTORS:
            padded_batch[key] = pad_sequence(key2values[key], batch_first=True)
        padded_batch["labels"] = torch.tensor(key2values["labels"], dtype=torch.float)
        return padded_batch