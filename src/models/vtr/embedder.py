import torch
from loguru import logger
from torch import nn
from torch.nn.functional import relu


def get_conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int | str = 0,
    stride: int = 1,
    dilation: int = 1,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        ),
        nn.BatchNorm2d(out_channels),
    )


def get_conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int | str = 0,
    stride: int = 1,
    dilation: int = 1,
):
    return nn.Sequential(
        get_conv_bn(in_channels, out_channels, kernel_size, padding, stride, dilation),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | str,
        stride: int,
        dilation: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = nn.Sequential(
            get_conv_bn_relu(in_channels, out_channels, kernel_size, padding, stride, dilation),
            get_conv_bn(out_channels, out_channels, kernel_size, padding, stride, dilation),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = relu(x)
        return x


def get_res_block_with_pooling(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int | str = 0,
    stride: int = 1,
    dilation: int = 1,
):
    return nn.Sequential(
        ResBlock(in_channels, out_channels, kernel_size, padding, stride, dilation),
        nn.MaxPool2d(2),
    )


class VisualEmbedder(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        kernel_size: int = 3,
        emb_size: int = 768,
        out_channels: int = 256,
        out_channels_res_block_1=64,
        out_channels_res_block_2=128,
    ):
        super().__init__()
        logger.info(f"Initializing VisualEmbedder | kernel size: {kernel_size}, emb size: {emb_size}")

        self.slice_conv = nn.Sequential(
            get_res_block_with_pooling(1, out_channels_res_block_1, kernel_size, padding="same"),
            get_res_block_with_pooling(
                out_channels_res_block_1,
                out_channels_res_block_2,
                kernel_size,
                padding="same",
            ),
            get_res_block_with_pooling(out_channels_res_block_2, out_channels, kernel_size, padding="same"),
        )

        self.linear_bridge = nn.Linear(
            (height // (2**3)) * (width // (2**3)) * out_channels,
            emb_size,
        )

    def forward(self, slices):
        batch_size, slice_count, height, width = slices.shape
        conv = self.slice_conv(slices.view(batch_size * slice_count, 1, height, width))

        _, channels_count, h_out, w_out = conv.shape

        batched_conv = conv.view(batch_size, slice_count, channels_count * h_out * w_out)
        return self.linear_bridge(batched_conv)  # [batch size, slice count, emb size]


class VisualEmbedderSL(VisualEmbedder):
    def __init__(
        self,
        height: int,
        width: int,
        kernel_size: int = 3,
        emb_size: int = 768,
        out_channels: int = 256,
        out_channels_res_block_1=64,
        out_channels_res_block_2=128,
    ):
        logger.info(f"Initializing VisualEmbedderSL")
        super().__init__(
            height, width, kernel_size, emb_size, out_channels, out_channels_res_block_1, out_channels_res_block_2
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        slices = batch["slices"]
        slice_embeddings = super().__call__(slices)  # [batch size, slice count, emb size]
        batch_size, slice_count, emb_size = slice_embeddings.shape

        masked_slice_embeddings = slice_embeddings * batch["tokens_mask"][:, :, None]

        max_word_len = int(batch["max_word_len"].item())
        masked_slice_embeddings_splitted_into_words = masked_slice_embeddings.view(
            batch_size, slice_count // max_word_len, max_word_len, emb_size
        )
        tokens_count_in_each_word = (
            batch["tokens_mask"].view(batch_size, slice_count // max_word_len, max_word_len).sum(dim=2)
        )
        word_embeddings = (
            torch.mean(masked_slice_embeddings_splitted_into_words, 2)
            / torch.max(torch.tensor(1), tokens_count_in_each_word)[:, :, None]
        )
        return word_embeddings
