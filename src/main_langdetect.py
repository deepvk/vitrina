import json
import pickle
from argparse import ArgumentParser, Namespace

from loguru import logger
from transformers import NllbTokenizer

from src.datasets.common import (
    AugmentationDataset,
    SlicesDataset,
    SlicesIterableDataset,
    TokenizedDataset,
    TokenizedIterableDataset,
)
from src.datasets.translation_datasets import NLLBDataset, FloresDataset
from src.models.embedders.ttr import TTREmbedder
from src.models.embedders.vtr import VTREmbedder
from src.models.tasks import SequenceClassifier
from src.utils.augmentation import init_augmentations
from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig, AugmentationConfig
from src.utils.train import train


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()

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

    arg_parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Augmentations are not applied to texts.",
    )

    arg_parser.add_argument("--dataset-dir", type=str, help="Directory of validation and test Flores files.")

    arg_parser.add_argument(
        "--tokenizer", type=str, default="resources/tokenizer", help="Path to tokenizer [only for vanilla model]."
    )

    arg_parser.add_argument("--vtr", action="store_true", help="Use Visual Token Representations.")

    arg_parser = VTRConfig.add_to_arg_parser(arg_parser)
    arg_parser = TransformerConfig.add_to_arg_parser(arg_parser)
    arg_parser = TrainingConfig.add_to_arg_parser(arg_parser)
    arg_parser = AugmentationConfig.add_to_arg_parser(arg_parser)
    return arg_parser


def train_langdetect(args: Namespace):
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    augmentation_config = AugmentationConfig.from_arguments(args)

    with open(args.probas, "rb") as f:
        probas = pickle.load(f)

    with open(args.char2array, "rb") as f:
        char2array = pickle.load(f)

    with open(augmentation_config.leet) as json_file:
        leet_symbols = json.load(json_file)

    with open(augmentation_config.clusters, "rb") as f:
        cluster_symbols = pickle.load(f)

    vtr = VTRConfig.from_arguments(args)
    channels = (1, 64, 128, vtr.out_channels)

    augmentations = init_augmentations(
        expected_changes_per_word=augmentation_config.expected_changes_per_word,
        cluster_symbols=cluster_symbols,
        leet_symbols=leet_symbols,
    )

    nllb_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-3.3B")

    train_dataset = NLLBDataset(probas=probas)
    lang2label = train_dataset.get_lang2label()

    if args.no_augmentation:
        val_dataset = FloresDataset(lang2label=lang2label, split="dev")
        test_dataset = FloresDataset(lang2label=lang2label, split="devtest")
    else:
        if not args.dataset_dir:
            logger.error("Need directory with augmented val and test Flores")
            return

        logger.info(
            f"Noisy dataset: expected_changes_per_word:{augmentation_config.expected_changes_per_word}, proba_per_text:{augmentation_config.proba_per_text}, expected_changes_per_text:{augmentation_config.expected_changes_per_text}, max_augmentations={augmentation_config.max_augmentations}"
        )

        train_dataset = AugmentationDataset(
            dataset=train_dataset,
            augmentations=augmentations,
            proba_per_text=augmentation_config.proba_per_text,
            expected_changes_per_text=augmentation_config.expected_changes_per_text,
            max_augmentations=augmentation_config.max_augmentations,
        )
        val_dataset = FloresDataset(lang2label, split="dev", dataset_dir=args.dataset_dir)
        test_dataset = FloresDataset(lang2label, split="devtest", dataset_dir=args.dataset_dir)

    if args.vtr:
        train_dataset = SlicesIterableDataset(train_dataset, char2array)
        val_dataset = SlicesDataset(val_dataset, char2array)
        test_dataset = SlicesDataset(test_dataset, char2array)

        embedder = VTREmbedder(
            height=vtr.font_size,
            width=vtr.window_size,
            conv_kernel_size=vtr.conv_kernel_size,
            pool_kernel_size=vtr.pool_kernel_size,
            emb_size=model_config.emb_size,
            channels=channels,
        )
    else:
        train_dataset = TokenizedIterableDataset(train_dataset, nllb_tokenizer, training_config.max_seq_len)
        val_dataset = TokenizedDataset(val_dataset, nllb_tokenizer, training_config.max_seq_len)
        test_dataset = TokenizedDataset(test_dataset, nllb_tokenizer, training_config.max_seq_len)

        embedder = TTREmbedder(train_dataset.tokenizer.vocab_size, model_config.emb_size)

    model_config.num_classes = train_dataset.get_num_classes()
    model = SequenceClassifier(model_config, embedder, training_config.max_seq_len)

    train(
        model,
        train_dataset,
        training_config,
        sl=False,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        ocr_flag=False,
        lang_detect_flag=True,
    )


if __name__ == "__main__":
    logger.info("Loading data...")
    _args = configure_arg_parser().parse_args()
    train_langdetect(_args)
