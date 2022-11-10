from typing import Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel

from datasets.bert_dataset_sl import BERTDatasetSL
from utils.utils import load_json


class EncoderForSequenceLabeling(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 768
        model_config = BertConfig(
            vocab_size=30_000,
            max_position_embeddings=512,
            hidden_size=hidden_size,
            num_attention_heads=12,
            num_hidden_layers=1,
            type_vocab_size=1,
        )
        self.bert = BertModel(model_config)
        self.word_classifier = nn.Linear(hidden_size, 1)

    def forward(self, batch: Dict[str, Union[torch.Tensor, int]]):
        bert_output = self.bert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        masked_output = (
            bert_output["last_hidden_state"] * batch["attention_mask"][:, :, None]
        )
        batch_size, seq_len, hidden_size = masked_output.shape
        word_len = batch["max_word_len"]
        word_split_output = masked_output.view(
            batch_size, seq_len // word_len, word_len, hidden_size
        )
        tokens_count_in_each_word = (
            batch["attention_mask"]
            .view(batch_size, seq_len // word_len, word_len)
            .sum(dim=2)
        )
        word_embeddings = (
            torch.mean(word_split_output, 2)
            / torch.max(torch.tensor(1), tokens_count_in_each_word)[:, :, None]
        )
        return self.word_classifier(word_embeddings).squeeze(2)


if __name__ == "__main__":
    model = EncoderForSequenceLabeling()
    labeled_texts = load_json("data/vk_toxic_sl.jsonl")
    dataset = BERTDatasetSL(labeled_texts, "berts/toxic-bert")
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_function)

    batch = next(iter(data_loader))

    print(batch)
    output = model(batch)
    print(output)
