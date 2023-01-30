import torch
from loguru import logger
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.datasets.common import DatasetSample
from src.utils.common import clean_text


class BERTDataset(Dataset):
    def __init__(
        self,
        labeled_texts: list[DatasetSample],
        tokenizer: str,
        max_seq_len: int = 512,
    ):
        logger.info(f"Initializing BERTDataset with {len(labeled_texts)} samples, use max seq len {max_seq_len}")
        self.labeled_texts = labeled_texts
        self.max_seq_len = max_seq_len

        logger.info(f"Loading tokenizer from '{tokenizer}'")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

    def __len__(self) -> int:
        return len(self.labeled_texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        labeled_text = self.labeled_texts[index]
        raw_text = clean_text(labeled_text["text"])
        label = labeled_text["label"]
        return raw_text, label

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
