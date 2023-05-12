import pickle
from loguru import logger
from torch.utils.data import Dataset
from argparse import ArgumentParser, Namespace

from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig
from src.utils.train import train
from src.utils.common import load_json
from src.datasets.vtr_dataset import VTRDataset, VTRDatasetOCR
from src.models.pretraining import MaskedVisualLM
from src.models.vtr.ocr import OCRHead


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--train-data", type=str, default=f"resources/data/train_dataset.jsonl", help="Path to train dataset."
    )
    arg_parser.add_argument("--val-data", type=str, default=None, help="Path to validation dataset.")
    arg_parser.add_argument("--test-data", type=str, default=None, help="Path to test dataset.")
    arg_parser.add_argument(
        "--char2array",
        type=str,
        default="resources/char2array.pkl",
        help="Path to char2array [only for VTR model].",
    )
    arg_parser.add_argument("--no-ocr", action="store_true", help="Do not use OCR with visual models.")

    arg_parser = VTRConfig.add_to_arg_parser(arg_parser)
    arg_parser = TransformerConfig.add_to_arg_parser(arg_parser)
    arg_parser = TrainingConfig.add_to_arg_parser(arg_parser)
    return arg_parser


def pretrain_vtr(args: Namespace, train_data: list, val_data: list = None, test_data: list = None):
    logger.info("Pre-training masked language model for VTR.")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    vtr = VTRConfig.from_arguments(args)

    with open(args.char2array, "rb") as f:
        char2array = pickle.load(f)

    dataset_args = (char2array, vtr.window_size, vtr.stride, training_config.max_seq_len)
    if args.no_ocr:
        train_dataset: Dataset = VTRDataset(train_data, *dataset_args)
        val_dataset: Dataset = VTRDataset(val_data, *dataset_args) if val_data else None
        test_dataset: Dataset = VTRDataset(test_data, *dataset_args) if test_data else None

        model = MaskedVisualLM(
            model_config.n_head,
            model_config.num_layers,
            model_config.dropout,
            vtr.font_size,
            vtr.window_size,
            vtr.verbose,
            vtr.save_plots,
        )
    else:
        train_dataset = VTRDatasetOCR(train_data, ratio=vtr.ratio, *dataset_args)
        val_dataset = VTRDatasetOCR(val_data, ratio=vtr.ratio, *dataset_args) if val_data else None
        test_dataset = VTRDatasetOCR(test_data, ratio=vtr.ratio, *dataset_args) if test_data else None

        char2int = {char: i + 1 for i, char in enumerate(char2array.keys())}

        logger.info(
            f"OCR parameters: hidden size: {vtr.hidden_size_ocr}, # layers: {vtr.num_layers_ocr}, "
            f"# classes: {len(char2array.keys())}"
        )
        ocr = OCRHead(
            input_size=vtr.font_size,
            hidden_size=vtr.hidden_size_ocr,
            num_layers=vtr.num_layers_ocr,
            num_classes=len(char2array.keys()),
        )

        model = MaskedVisualLM(
            model_config.n_head,
            model_config.num_layers,
            model_config.dropout,
            vtr.font_size,
            vtr.window_size,
            vtr.verbose,
            vtr.save_plots,
            ocr,
            char2int,
            vtr.alpha,
        )

    train(
        model,
        train_dataset,
        training_config,
        sl=False,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        ocr_flag=not args.no_ocr,
    )


def main(args: Namespace):
    logger.info("Loading data...")
    train_data = load_json(args.train_data)
    val_data = load_json(args.val_data) if args.val_data else None
    test_data = load_json(args.test_data) if args.test_data else None

    pretrain_vtr(args, train_data, val_data, test_data)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
