from torch import nn


class TTREmbedder(nn.Module):
    def __init__(self, num_embeddings, emb_size):
        super().__init__()

        self.embedder = nn.Embedding(num_embeddings, emb_size)

    def forward(self, batch):
        output = {"embeddings": self.embedder(batch["input_ids"])}

        return output
