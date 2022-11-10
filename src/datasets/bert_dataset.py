from collections import defaultdict
from typing import List, Dict, Union, Any

import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.utils import clean_text


class BERTDataset(Dataset):
    PADDED_VECTORS = ["input_ids", "attention_mask", "token_type_ids"]

    def __init__(self, labeled_texts: List[Dict[str, Union[str, int]]], tokenizer: str, max_seq_len: int = 512):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.labeled_texts = labeled_texts
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labeled_texts)

    def __getitem__(self, index):
        labeled_text = self.labeled_texts[index]

        encoded_dict = self.tokenizer(
            clean_text(labeled_text["text"]),
            add_special_tokens=True,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        for key in BERTDataset.PADDED_VECTORS:
            encoded_dict[key] = encoded_dict[key].squeeze(0)
        encoded_dict["labels"] = labeled_text["label"]

        return encoded_dict

    @staticmethod
    def collate_function(batch: List[Dict[str, Union[torch.Tensor, int]]]) -> Dict[str, torch.Tensor]:
        key2values = defaultdict(list)
        for item in batch:
            for key, val in item.items():
                key2values[key].append(val)

        padded_batch = {}
        for key in BERTDataset.PADDED_VECTORS:
            padded_batch[key] = pad_sequence(key2values[key], batch_first=True)
        padded_batch["labels"] = torch.tensor(key2values["labels"], dtype=torch.float)
        return padded_batch
