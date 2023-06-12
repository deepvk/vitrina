import argparse
import pickle

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def pad_image(input_array, width, height):
    padded_array = np.zeros((height, width))
    shape = np.shape(input_array)
    shift = (width - shape[1]) // 2
    padded_array[: shape[0], shift : shape[1] + shift] = input_array
    return padded_array


def clusterizer(
    n_clusters: int,
    char2array: str,
    save_to: str,
):
    with open(char2array, "rb") as f:
        char2array = pickle.load(f)

    max_width = max([v.shape[1] for k, v in char2array.items()])
    height = char2array["a"].shape[0]

    char2array_padded = {}
    for k, v in char2array.items():
        char2array_padded[k] = pad_image(v, max_width, height)

    letters = [symbol for symbol in char2array_padded.keys() if symbol.isalpha()]
    letters_dict = {letter: char2array_padded[letter] for letter in letters}

    x_train = np.array(list(letters_dict.values()))

    x_train = x_train / 255.0
    X_train = x_train.reshape(len(x_train), -1)

    clusterizer = MiniBatchKMeans(n_clusters=n_clusters)
    clusters = clusterizer.fit_predict(X_train)

    char2cluster = {ch: cluster for ch, cluster in zip(letters_dict.keys(), clusters)}
    cluster2char = [[] for _ in range(clusterizer.n_clusters)]
    for ch, cluster in char2cluster.items():
        cluster2char[cluster].append(ch)

    result = {ch: cluster2char[char2cluster[ch]] for ch in letters}

    with open(save_to, "wb") as json_file:
        pickle.dump(result, json_file)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=4000)
    parser.add_argument("--save-to", type=str, default=f"resources/nllb/letter_replacement/clusterization.pkl")
    parser.add_argument("--char2array", type=str, default="resources/char2array.pkl")
    args = parser.parse_args()
    clusterizer(**vars(args))
