from torch import nn


class TTREmbedder(nn.Module):
    def __init__(self, num_embeddings, emb_size):
        super().__init__()

    def forward(self, batch):
        output = {"embeddings": [batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]]}
        return output
