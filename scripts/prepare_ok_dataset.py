from argparse import ArgumentParser

from utils.utils import save_json


def main(data: str, save_to: str):
    result = []

    with open(data, "r") as file:
        for line in file:
            labels = line.split()[0]
            text = line[len(labels) + 1 :].strip()
            labels = labels.split(",")
            result.append({"text": text, "toxic": int("__label__NORMAL" not in labels)})
    save_json(result, save_to)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="resources/data/dataset.txt")
    parser.add_argument("--save-to", type=str, default="resources/data/ok_toxic.jsonl")
    args = parser.parse_args()
    main(args.data, args.save_to)
