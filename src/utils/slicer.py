import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from src.utils.common import text2image


class VTRSlicerWithText:
    def __init__(
        self,
        window_size: int = 25,
        stride: int = 10,
        font: str = "fonts/NotoSans.ttf",
        font_size: int = 15,
        ratio: float = 0.7,
    ):
        logger.info(
            f"Init VTRSlicerWithText | window_size={window_size}, stride={stride}, "
            f"font={font}, font_size={font_size}, ratio={ratio}"
        )
        self.window_size = window_size
        self.stride = stride
        self.font = font
        self.font_size = font_size
        self.ratio = ratio

    def __call__(self, text: str, max_slice_count: int = None) -> tuple[torch.Tensor, list[str]]:
        image = []
        char_num = []
        char_ratio_l = np.array([])
        char_ratio_r = np.array([])

        if len(text) == 0:
            text = " "

        for i in range(len(text)):
            char_img = text2image(text[i], font=self.font, font_size=self.font_size)
            width = char_img.size[0]
            char_num += [i] * width
            char_ratio_l = np.concatenate((char_ratio_l, np.arange(1, 0, -1 / width)))
            char_ratio_r = np.concatenate((char_ratio_r, np.arange(1 / width, 1 + 1 / width, 1 / width)))
            image.append(char_img)

        image_bytes = torch.as_tensor(np.hstack(image)).float()

        image_width = image_bytes.shape[1]
        padded_image_width = int(
            ((max(image_width, self.window_size) - self.window_size + self.stride - 1) // self.stride) * self.stride
            + self.window_size
        )

        image_bytes = F.pad(image_bytes, (0, padded_image_width - image_width), "constant", 255)
        image_bytes = 255 - image_bytes
        slices = image_bytes.unfold(1, self.window_size, self.stride).permute((1, 0, 2))[:max_slice_count]

        lb = 0
        rb = min(self.window_size - 1, len(char_num) - 1)
        r_shift = 1 if char_ratio_r[rb] >= self.ratio else 0
        slice_text = [text[char_num[lb] : char_num[rb] + r_shift]]

        for i in range(len(slices) - 1):
            lb += self.stride
            rb += self.stride
            if rb >= len(char_num):
                rb = len(char_num) - 1

            l_shift = 0 if char_ratio_l[lb] >= self.ratio else 1
            r_shift = 1 if char_ratio_r[rb] >= self.ratio else 0
            slice_text.append(text[char_num[lb] + l_shift : char_num[rb] + r_shift])

        return slices, slice_text


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
