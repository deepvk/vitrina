import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from src.utils.common import text2image


class VTRSlicer:
    def __init__(
        self,
        window_size: int = 25,
        stride: int = 10,
        font: str = "fonts/NotoSans.ttf",
        font_size: int = 15,
    ):
        logger.info(f"Init VTRSlicer | window_size={window_size}, stride={stride}, font={font}, font_size={font_size}")
        self.window_size = window_size
        self.stride = stride
        self.font = font
        self.font_size = font_size

    def __call__(self, text: str, max_slice_count: int = None) -> torch.Tensor:
        image = text2image(text, font=self.font, font_size=self.font_size)
        image_bytes = torch.as_tensor(np.array(image))

        image_width = image_bytes.shape[1]
        padded_image_width = int(
            ((max(image_width, self.window_size) - self.window_size + self.stride - 1) // self.stride) * self.stride
            + self.window_size
        )

        image_bytes = F.pad(image_bytes, (0, padded_image_width - image_width), "constant", 255)
        image_bytes = 255 - image_bytes
        slices = image_bytes.unfold(1, self.window_size, self.stride).permute((1, 0, 2))[:max_slice_count]

        return slices
