import argparse
import glob
import json
import os

import freetype
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from sklearn.cluster import MiniBatchKMeans

RUSSIAN_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
FONT_COLOR = "#000000"
BACKGROUND_COLOR = "#FFFFFF"


def clusterization(
    font_path: str = "resources/fonts/NotoSans.ttf",
    output_path: str = "resources/letter_replacement/clusterization.json",
    font_size: int = 13,
    clusters: int = 500,
) -> dict[str, list[str]]:
    """
    Clustering character images supported by a specified font.

    :param font_path: the path to the font that needs to generate images (default: fonts/NotoSans.ttf).
    :param output_path: the path to the output file (default: letter_replacement/clusterization.json).
    :param font_size: font size in pixels (default: 13)
    :param clusters: number of clusters into which to divide a set of character images (default: 500)
    :return: a dictionary {russian symbol: a list of symbols from the same cluster}
    """
    face = freetype.Face(font_path)
    image_font = ImageFont.truetype(font_path, max(font_size - 2, 8))

    supported_chars = sorted([chr(c) for c, g in face.get_chars() if c])
    max_len = 0
    for c in supported_chars:
        line_width, line_height = image_font.getsize(c)
        max_len = max(max_len, line_height, line_width)

    letter_images_dirname = "letter_images"
    all_images_dirname = os.path.join(letter_images_dirname, "all")
    if not os.path.exists(all_images_dirname):
        os.makedirs(all_images_dirname)

    char2image = {}
    for ind, c in enumerate(supported_chars):
        image = Image.new("L", (max_len, max_len), color=BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)
        w, h = draw.textsize(c, font=image_font)
        draw.text(
            xy=((max_len - w) / 2, (max_len - h) / 2),
            text=c,
            fill=FONT_COLOR,
            font=image_font,
        )
        image.save(f"{all_images_dirname}/{ind}.jpg")
        char2image[c] = image

    char2vector = {ch: np.asarray(image).reshape(-1).astype(np.float32) / 255.0 for ch, image in char2image.items()}

    clusterizer = MiniBatchKMeans(n_clusters=clusters)
    clusters = clusterizer.fit_predict(list(char2vector.values()))

    char2cluster = {ch: cluster for ch, cluster in zip(char2image.keys(), clusters)}
    cluster2char = [[] for _ in range(clusterizer.n_clusters)]

    for ch, cluster in char2cluster.items():
        cluster2char[cluster].append(ch)

    for lower_letter in RUSSIAN_LETTERS:
        for is_upper, letter in enumerate([lower_letter, lower_letter.upper()]):
            cluster = char2cluster[letter]
            letter_name = f"capital_{lower_letter}" if is_upper else lower_letter
            letter_dir = f"{letter_images_dirname}/{letter_name}"
            if not os.path.exists(letter_dir):
                os.mkdir(letter_dir)
            else:
                for file in glob.glob(letter_dir + "/*"):
                    os.remove(file)

            for ind, similar_letter in enumerate(cluster2char[cluster]):
                char2image[similar_letter].save(f"{letter_dir}/{similar_letter}.jpg")

    result = {ch: cluster2char[char2cluster[ch]] for ch in RUSSIAN_LETTERS + RUSSIAN_LETTERS.upper()}
    with open(output_path, "w") as json_file:
        json.dump(result, json_file, ensure_ascii=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-path", type=str, default="resources/fonts/NotoSans.ttf")
    parser.add_argument("--output-path", type=str, default="resources/letter_replacement/clusterization.json")
    parser.add_argument("--font-size", type=int, default=13)
    parser.add_argument("--clusters", type=int, default=500)
    args = parser.parse_args()

    clusterization(**vars(args))
