from collections import defaultdict
from typing import List, Dict, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from datasets.sized_collated_dataset import SizedCollatedDataset
from utils.utils import clean_text, load_json


class BERTDatasetSL(SizedCollatedDataset[Union[List[List[Union[str, int]]], int]]):
    PADDED_VECTORS = ["input_ids"]

    def __init__(
        self,
        labeled_texts: List[Dict[str, Union[List[List[Union[str, int]]], int]]],
        tokenizer: str,
        max_seq_len: int = 512,
    ):
        super().__init__(labeled_texts)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labeled_texts)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        labeled_text = self.labeled_texts[index]["text"]

        encoded_words = []
        labels = []

        for word, label in labeled_text:
            cleaned_word = clean_text(word)
            if not cleaned_word:
                continue
            encoded_dict = self.tokenizer(
                cleaned_word,
                add_special_tokens=False,
                max_length=self.max_seq_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
                return_tensors="pt",
            )
            encoded_word = encoded_dict["input_ids"][0]
            encoded_words.append(encoded_word)
            labels.append(label)

        return {"words_input_ids": torch.cat(encoded_words), "labels": torch.tensor(labels)}

    def collate_function(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        key2values = defaultdict(list)
        max_seq_len = 0
        max_word_len = 0
        for item in batch:
            word_input_ids = item["words_input_ids"]
            max_seq_len = max(max_seq_len, len(word_input_ids))
            for word in word_input_ids:
                max_word_len = max(max_word_len, len(word))
            key2values["words_input_ids"].append(item["words_input_ids"])

            key2values["labels"].append(torch.tensor(item["labels"]))

        pad_token_id = self.tokenizer.pad_token_id

        max_seq_len_in_words = self.max_seq_len // max_word_len
        max_seq_len_in_tokens = max_seq_len_in_words * max_word_len

        batch_labels = []
        for text, labels in zip(key2values["words_input_ids"], key2values["labels"]):
            text_attention_mask = []
            words = []
            for word in text:
                padding_size = max_word_len - len(word)
                words.append(
                    F.pad(
                        word,
                        (0, padding_size),
                        mode="constant",
                        value=pad_token_id,
                    )
                )
                text_attention_mask.extend([1] * len(word) + [0] * padding_size)

            if len(words) > 0:
                key2values["input_ids"].append(torch.cat(words)[:max_seq_len_in_tokens])
                key2values["attention_mask"].append(torch.tensor(text_attention_mask)[:max_seq_len_in_tokens])
                batch_labels.append(labels[:max_seq_len_in_words])

        return {
            "max_word_len": torch.tensor(max_word_len, dtype=torch.int32),
            "input_ids": torch.nn.utils.rnn.pad_sequence(key2values["input_ids"], batch_first=True).long(),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(key2values["attention_mask"], batch_first=True).long(),
            "labels": torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-1).long(),
        }


if __name__ == "__main__":
    labeled_texts = load_json("data/vk_toxic_sl.jsonl")
    dataset = BERTDatasetSL(labeled_texts, "berts/toxic-bert")
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_function)
    print(next(iter(data_loader)))
