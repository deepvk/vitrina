from torch import nn


class OCRHead(nn.Module):
    def __init__(self, input_size=256, hidden_size=1024, num_layers=2, num_classes=44):
        super(OCRHead, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, slices_emb):
        slices, channel, height, width = slices_emb.size()

        slices_emb = slices_emb.view(slices, channel * height, width)
        slices_emb = slices_emb.permute(2, 0, 1)

        out, _ = self.lstm(slices_emb)
        out = self.fc(out)

        return out
