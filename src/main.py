from argparse import ArgumentParser, Namespace

from loguru import logger
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import Dataset

from src.datasets.bert_dataset import BERTDataset
from src.datasets.bert_dataset_sl import BERTDatasetSL
from src.datasets.vtr_dataset import VTRDataset, VTRDatasetOCR
from src.datasets.vtr_dataset_sl import VTRDatasetSL
from src.models.ttr.classifier import TokensToxicClassifier
from src.models.ttr.sequence_labeler import TextTokensSequenceLabeler
from src.models.vtr.classifier import VisualToxicClassifier
from src.models.vtr.sequence_labeler import VisualTextSequenceLabeler
from src.utils.common import load_json, BceLossForTokenClassification
from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig
from src.utils.train import train


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--train-data", type=str, default=f"resources/data/train_dataset.jsonl", help="Path to train dataset."
    )
    arg_parser.add_argument("--val-data", type=str, default=None, help="Path to validation dataset.")
    arg_parser.add_argument("--test-data", type=str, default=None, help="Path to test dataset.")

    arg_parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer [only for vanilla model].")

    arg_parser.add_argument("--vtr", action="store_true", help="Use Visual Token Representations.")
    arg_parser.add_argument("--sl", action="store_true", help="Use Sequence Labeling task.")

    arg_parser.add_argument("--no-ocr", action="store_true", help="Do not use OCR with visual models.")

    arg_parser = VTRConfig.add_to_arg_parser(arg_parser)
    arg_parser = TransformerConfig.add_to_arg_parser(arg_parser)
    arg_parser = TrainingConfig.add_to_arg_parser(arg_parser)
    return arg_parser


def train_vanilla_encoder(args: Namespace, train_data: list, val_data: list = None, test_data: list = None):
    logger.info("Training Vanilla Encoder for sequence classification.")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)

    train_dataset = BERTDataset(train_data, args.tokenizer, training_config.max_seq_len)
    val_dataset = BERTDataset(val_data, args.tokenizer, training_config.max_seq_len) if val_data else None
    test_dataset = BERTDataset(test_data, args.tokenizer, training_config.max_seq_len) if test_data else None

    model = TokensToxicClassifier(
        vocab_size=train_dataset.tokenizer.vocab_size,
        num_layers=model_config.num_layers,
        hidden_size=model_config.emb_size,
        num_attention_heads=model_config.n_head,
        dropout=model_config.dropout,
    )
    criterion = BCEWithLogitsLoss()

    train(
        model, train_dataset, criterion, training_config, sl=False, val_dataset=val_dataset, test_dataset=test_dataset
    )


def train_vanilla_encoder_sl(args: Namespace, train_data: list, val_data: list = None, test_data: list = None):
    logger.info("Training Vanilla Encoder for sequence labeling.")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)

    train_dataset = BERTDatasetSL(train_data, args.tokenizer, training_config.max_seq_len)
    val_dataset = BERTDatasetSL(val_data, args.tokenizer, training_config.max_seq_len) if val_data else None
    test_dataset = BERTDatasetSL(test_data, args.tokenizer, training_config.max_seq_len) if test_data else None

    model = TextTokensSequenceLabeler(
        vocab_size=train_dataset.tokenizer.vocab_size,
        num_layers=model_config.num_layers,
        hidden_size=model_config.emb_size,
        num_attention_heads=model_config.n_head,
        dropout=model_config.dropout,
    )
    criterion = BceLossForTokenClassification()

    train(model, train_dataset, criterion, training_config, sl=True, val_dataset=val_dataset, test_dataset=test_dataset)


def train_vtr_encoder(args: Namespace, train_data: list, val_data: list = None, test_data: list = None):
    logger.info("Training Visual Token Representation Encoder for sequence classification.")
    logger.info(f"OCR: {not args.no_ocr}")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    vtr = VTRConfig.from_arguments(args)

    model = VisualToxicClassifier(
        height=vtr.font_size,
        width=vtr.window_size,
        kernel_size=vtr.kernel_size,
        out_channels=vtr.out_channels,
        num_layers=model_config.num_layers,
        hidden_size=model_config.emb_size,
        num_attention_heads=model_config.n_head,
        num_classes=model_config.num_classes,
        dropout=model_config.dropout,
    )
    criterion = CrossEntropyLoss()

    if args.no_ocr:
        train_dataset: Dataset = VTRDataset(
            train_data, vtr.font, vtr.font_size, vtr.window_size, vtr.stride, training_config.max_seq_len
        )

        val_dataset: Dataset = (
            VTRDataset(val_data, vtr.font, vtr.font_size, vtr.window_size, vtr.stride, training_config.max_seq_len)
            if val_data
            else None
        )

        test_dataset: Dataset = (
            VTRDataset(test_data, vtr.font, vtr.font_size, vtr.window_size, vtr.stride, training_config.max_seq_len)
            if test_data
            else None
        )

    else:
        train_dataset = VTRDatasetOCR(
            train_data, vtr.font, vtr.font_size, vtr.window_size, vtr.stride, training_config.max_seq_len, vtr.ratio
        )

        val_dataset = (
            VTRDatasetOCR(
                val_data, vtr.font, vtr.font_size, vtr.window_size, vtr.stride, training_config.max_seq_len, vtr.ratio
            )
            if val_data
            else None
        )

        test_dataset = (
            VTRDatasetOCR(
                test_data, vtr.font, vtr.font_size, vtr.window_size, vtr.stride, training_config.max_seq_len, vtr.ratio
            )
            if test_data
            else None
        )

    train(
        model, train_dataset, criterion, training_config, sl=False, val_dataset=val_dataset, test_dataset=test_dataset
    )


def train_vtr_encoder_sl(args: Namespace, train_data: list, val_data: list = None, test_data: list = None):
    logger.info("Training Visual Token Representation Encoder for sequence labeling.")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    vtr = VTRConfig.from_arguments(args)

    model = VisualTextSequenceLabeler(
        height=vtr.font_size,
        width=vtr.window_size,
        kernel_size=vtr.kernel_size,
        out_channels=vtr.out_channels,
        emb_size=model_config.emb_size,
        num_layers=model_config.num_layers,
        n_heads=model_config.n_head,
        dropout=model_config.dropout,
    )
    criterion = BceLossForTokenClassification()

    train_dataset = VTRDatasetSL(
        train_data, vtr.font, vtr.font_size, vtr.window_size, vtr.stride, training_config.max_seq_len
    )
    val_dataset = (
        VTRDatasetSL(val_data, vtr.font, vtr.font_size, vtr.window_size, vtr.stride, training_config.max_seq_len)
        if val_data
        else None
    )
    test_dataset = (
        VTRDatasetSL(test_data, vtr.font, vtr.font_size, vtr.window_size, vtr.stride, training_config.max_seq_len)
        if test_data
        else None
    )

    train(
        model, train_dataset, criterion, training_config, sl=False, val_dataset=val_dataset, test_dataset=test_dataset
    )


def main(args: Namespace):
    if not args.vtr and not args.tokenizer:
        logger.error("You should specify tokenizer path for vanilla model.")
        return

    logger.info("Loading data...")
    train_data = load_json(args.train_data)
    val_data = load_json(args.val_data) if args.val_data else None
    test_data = load_json(args.test_data) if args.test_data else None

    if args.vtr and args.sl:
        train_vtr_encoder_sl(args, train_data, val_data, test_data)
    elif args.vtr:
        train_vtr_encoder(args, train_data, val_data, test_data)
    elif args.sl:
        train_vanilla_encoder_sl(args, train_data, val_data, test_data)
    else:
        train_vanilla_encoder(args, train_data, val_data, test_data)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
