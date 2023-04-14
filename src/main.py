import pickle
from argparse import ArgumentParser, Namespace

from loguru import logger
from torch.utils.data import Dataset, IterableDataset

from src.datasets.bert_dataset import BERTDataset
from src.datasets.nllb_dataset import DatasetNLLB, FloresDataset
from src.datasets.bert_dataset_sl import BERTDatasetSL
from src.datasets.vtr_dataset import VTRDataset, VTRDatasetOCR
from src.datasets.vtr_dataset_sl import VTRDatasetSL
from src.models.ttr.sequence_labeler import TextTokensSequenceLabeler
from src.models.vtr.sequence_labeler import VisualTextSequenceLabeler
from src.utils.common import load_json
from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig
from src.utils.train import train
from src.models.embedders.vtr import VTREmbedder
from src.models.embedders.ttr import TTREmbedder
from src.models.vtr.ocr import OCRHead
from src.models.tasks import SequenceClassifier


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

    arg_parser.add_argument(
        "--char2array",
        type=str,
        default="resources/char2array.pkl",
        help="Path to char2array [only for VTR model].",
    )

    arg_parser.add_argument(
        "--probas",
        type=str,
        default="resources/nllb/probas_nllb.pkl",
        help="Path to probabilities of language pairs [for lang detect task].",
    )

    arg_parser.add_argument("--no-ocr", action="store_true", help="Do not use OCR with visual models.")
    arg_parser.add_argument("--lang-detect", action="store_true", help="Use iterable dataset for language detection.")

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

    embedder = TTREmbedder(train_dataset.tokenizer.vocab_size, model_config.emb_size)

    model = SequenceClassifier(model_config, embedder, training_config.max_seq_len)

    train(model, train_dataset, training_config, sl=False, val_dataset=val_dataset, test_dataset=test_dataset)


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

    train(model, train_dataset, training_config, sl=True, val_dataset=val_dataset, test_dataset=test_dataset)


def train_vtr_encoder(args: Namespace, train_data: list = None, val_data: list = None, test_data: list = None):
    logger.info("Training Visual Token Representation Encoder for sequence classification.")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    vtr = VTRConfig.from_arguments(args)
    channels = (1, 64, 128, vtr.out_channels)

    embedder = VTREmbedder(
        height=vtr.font_size,
        width=vtr.window_size,
        conv_kernel_size=vtr.conv_kernel_size,
        pool_kernel_size=vtr.pool_kernel_size,
        emb_size=model_config.emb_size,
        channels=channels,
    )

    with open(args.char2array, "rb") as f:
        char2array = pickle.load(f)

    dataset_args = (char2array, vtr.window_size, vtr.stride, training_config.max_seq_len)

    if args.lang_detect:
        with open(args.probas, "rb") as f:
            probas = pickle.load(f)

        train_dataset = DatasetNLLB(char2array, probas, vtr.window_size, vtr.stride, training_config.max_seq_len)
        lang2label = train_dataset.get_lang2label()
        val_dataset = FloresDataset(
            lang2label, char2array, vtr.window_size, vtr.stride, training_config.max_seq_len, "dev"
        )
        test_dataset = FloresDataset(
            lang2label, char2array, vtr.window_size, vtr.stride, training_config.max_seq_len, "devtest"
        )

        model_config.num_classes = train_dataset.get_num_classes()

        model = SequenceClassifier(model_config, embedder, training_config.max_seq_len)

    elif args.no_ocr:
        train_dataset: Dataset = VTRDataset(train_data, *dataset_args)
        val_dataset: Dataset = VTRDataset(val_data, *dataset_args) if val_data else None
        test_dataset: Dataset = VTRDataset(test_data, *dataset_args) if test_data else None

        model = SequenceClassifier(model_config, embedder, training_config.max_seq_len)

    else:
        train_dataset = VTRDatasetOCR(train_data, ratio=vtr.ratio, *dataset_args)
        val_dataset = VTRDatasetOCR(val_data, ratio=vtr.ratio, *dataset_args) if val_data else None
        test_dataset = VTRDatasetOCR(test_data, ratio=vtr.ratio, *dataset_args) if test_data else None

        char2int_dict = {char: i + 1 for i, char in enumerate(char2array.keys())}

        logger.info(
            f"OCR parameters: hidden size: {vtr.hidden_size_ocr}, # layers: {vtr.num_layers_ocr}, "
            f"# classes: {len(char2array.keys())}"
        )
        ocr = OCRHead(
            input_size=vtr.out_channels * (vtr.font_size // vtr.pool_kernel_size ** (len(channels) - 1)),
            hidden_size=vtr.hidden_size_ocr,
            num_layers=vtr.num_layers_ocr,
            num_classes=len(char2array.keys()),
        )

        model = SequenceClassifier(model_config, embedder, training_config.max_seq_len, char2int_dict, ocr, vtr.alpha)

    train(
        model,
        train_dataset,
        training_config,
        sl=False,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        ocr_flag=not args.no_ocr,
        lang_detect_flag=args.lang_detect,
    )


def train_vtr_encoder_sl(args: Namespace, train_data: list, val_data: list = None, test_data: list = None):
    logger.info("Training Visual Token Representation Encoder for sequence labeling.")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    vtr = VTRConfig.from_arguments(args)

    model = VisualTextSequenceLabeler(
        height=vtr.font_size,
        width=vtr.window_size,
        kernel_size=vtr.conv_kernel_size,
        channels=(1, 64, 128, vtr.out_channels),
        emb_size=model_config.emb_size,
        num_layers=model_config.num_layers,
        n_heads=model_config.n_head,
        dropout=model_config.dropout,
    )

    with open(args.char2array, "rb") as f:
        char2array = pickle.load(f)

    dataset_args = (char2array, vtr.window_size, vtr.stride, training_config.max_seq_len)
    train_dataset = VTRDatasetSL(train_data, *dataset_args)
    val_dataset = VTRDatasetSL(val_data, *dataset_args) if val_data else None
    test_dataset = VTRDatasetSL(test_data, *dataset_args) if test_data else None

    train(model, train_dataset, training_config, sl=True, val_dataset=val_dataset, test_dataset=test_dataset)


def main(args: Namespace):
    if not args.vtr and not args.tokenizer and not args.lang_detect:
        logger.error("You should specify tokenizer path for vanilla model.")
        return

    logger.info("Loading data...")

    train_data = load_json(args.train_data) if not args.lang_detect else None
    val_data = load_json(args.val_data) if args.val_data else None
    test_data = load_json(args.test_data) if args.test_data else None

    if args.lang_detect:
        train_vtr_encoder(args)
    elif args.vtr and args.sl:
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
