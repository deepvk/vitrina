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
    conv_kernel_size: int,
    pool_kernel_size: int,
    padding: int | str = 0,
    stride: int = 1,
    dilation: int = 1,
):
    return nn.Sequential(
        ResBlock(in_channels, out_channels, conv_kernel_size, padding, stride, dilation),
        nn.MaxPool2d(pool_kernel_size),
    )


class VTREmbedder(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        emb_size: int = 768,
        channels: tuple = (1, 64, 128, 256),
    ):
        super().__init__()
        logger.info(
            f"Initializing VisualEmbedder | number of layers: {len(channels)-1}, emb size: {emb_size}, "
            f"convolutional kernel size: {conv_kernel_size}, pooling kernel size: {pool_kernel_size}"
        )

        layers = [
            get_res_block_with_pooling(channels[i], channels[i + 1], conv_kernel_size, pool_kernel_size, padding="same")
            for i in range(len(channels) - 1)
        ]
        self.slice_conv = nn.Sequential(*layers)

        self.linear_bridge = nn.Linear(
            (height // (pool_kernel_size ** (len(channels) - 1)))
            * (width // (pool_kernel_size ** (len(channels) - 1)))
            * channels[-1],
            emb_size,
        )

    def forward(self, batch):
        batch_size, slice_count, height, width = batch["slices"].shape
        conv = self.slice_conv(batch["slices"].view(batch_size * slice_count, 1, height, width))

        _, channels_count, h_out, w_out = conv.shape
        batched_conv = conv.view(batch_size, slice_count, channels_count * h_out * w_out)
        batched_conv = self.linear_bridge(batched_conv)

        return {
            "embeddings": batched_conv,  # [batch size, slice count, emb size],
            "ocr_embeddings": conv,  # [batch size * slice count, out channels, emb height, emb width]
        }
