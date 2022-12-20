from argparse import ArgumentParser

from src.utils.common import save_json


def prepare_ok_dataset(
    data: str = "resources/data/dataset.txt",
    save_to: str = "resources/data/ok_toxic.jsonl",
):
    """
    Converts the dataset of classmates to the format that the model works with.
    The output file has jsonl resolution and all labels in the result have a binary value.

    :param data: path to source dataset (default: resources/data/dataset.txt)
    :param save_to: the file where the result will be saved (default: resources/data/ok_toxic.jsonl)
    :return:
    """
    result = []

    with open(data, "r") as file:
        for line in file:
            labels = line.split()[0]
            text = line[len(labels) + 1 :].strip()
            labels = labels.split(",")
            result.append({"text": text, "label": int("__label__NORMAL" not in labels)})
    save_json(result, save_to)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="resources/data/dataset.txt")
    parser.add_argument("--save-to", type=str, default="resources/data/ok_toxic.jsonl")
    args = parser.parse_args()
    prepare_ok_dataset(args.data, args.save_to)
