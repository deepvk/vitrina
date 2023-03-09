import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


class VTRSlicerWithText:
    def __init__(
        self,
        char2array: dict,
        window_size: int = 32,
        stride: int = 5,
        ratio: float = 0.7,
    ):
        logger.info(f"Init VTRSlicerWithText | window_size={window_size}, stride={stride}, ratio={ratio}")
        self.window_size = window_size
        self.stride = stride
        self.ratio = ratio
        self.char2array = char2array
        self.unknown_token = char2array["UNK"]

    def __call__(self, text: str, max_slice_count: int = None) -> tuple[torch.Tensor, list[str]]:
        image = []
        char_num = []
        char_ratio_l = np.array([])
        char_ratio_r = np.array([])

        if len(text) == 0:
            text = " "

        for i in range(len(text)):
            char_img = self.char2array.get(text[i], self.unknown_token)
            width = char_img.shape[1]
            if width == 0:
                continue
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

        image_bytes = F.pad(image_bytes, (0, padded_image_width - image_width), "constant", 0)
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

        with open("slice_text.txt", "w", encoding="utf-8") as file:
            for txt in slice_text:
                file.write(txt + "\n")
        torch.save(slices, "slices.pt")

        return slices, slice_text


class VTRSlicer:
    def __init__(
        self,
        char2array: dict,
        window_size: int = 32,
        stride: int = 5,
    ):
        logger.info(f"Init VTRSlicer | window_size={window_size}")
        self.window_size = window_size
        self.stride = stride
        self.char2array = char2array
        self.unknown_token = char2array["UNK"]

    def __call__(self, text: str, max_slice_count: int = None) -> torch.Tensor:
        image_bytes = torch.tensor(
            np.concatenate([self.char2array.get(char, self.unknown_token) for char in text], axis=1)
        )

        image_width = image_bytes.shape[1]
        padded_image_width = int(
            ((max(image_width, self.window_size) - self.window_size + self.stride - 1) // self.stride) * self.stride
            + self.window_size
        )

        image_bytes = F.pad(image_bytes, (0, padded_image_width - image_width), "constant", 0)
        slices = image_bytes.unfold(1, self.window_size, self.stride).permute((1, 0, 2))[:max_slice_count]

        return slices
