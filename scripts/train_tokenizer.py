import argparse
from pathlib import Path

from tokenizers.implementations import BertWordPieceTokenizer

from src.utils.common import load_json


def train_tokenizer(data_path: str = "resources/data/noisy_dataset.jsonl", save_to: str = "tokenizer"):
    """
    Trains the WordPiece tokenizer. Used to operate a classic transformer encoder.

    :param data_path: the data on which the tokenizer is trained (default: resources/data/noisy_dataset.jsonl)
    :param save_to: where to save the tokenizer (default: tokenizer)
    :return:
    """
    bert_tokenizer = BertWordPieceTokenizer(
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        clean_text=False,
        handle_chinese_chars=False,
        lowercase=False,
        wordpieces_prefix="##",
    )

    data = load_json(data_path)
    texts = list(map(lambda x: x["text"], data))

    bert_tokenizer.train_from_iterator(
        texts,
        vocab_size=30_000,
        wordpieces_prefix="##",
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    output_dir = Path(save_to)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    bert_tokenizer.save_model(save_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=f"resources/data/noisy_dataset.jsonl")
    parser.add_argument("--save-to", type=str, default=f"tokenizer")

    args = parser.parse_args()
    train_tokenizer(data_path=args.data, save_to=args.save_to)
