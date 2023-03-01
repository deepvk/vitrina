from torch import nn


class VanillaEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        output = {"embeddings": [batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]]}
        return output
