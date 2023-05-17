import json
import pickle
from argparse import ArgumentParser, Namespace

from loguru import logger
from torch.utils.data import Dataset
from transformers import NllbTokenizer

from src.datasets.bert_dataset import BERTDataset
from src.datasets.common import AugmentationDataset, SlicesDataset
from src.datasets.translation_datasets import NLLBDataset, FloresDataset
from src.datasets.vtr_dataset import VTRDataset
from src.models.embedders.ttr import TTREmbedder
from src.models.embedders.vtr import VTREmbedder
from src.models.tasks import SequenceClassifier
from src.utils.augmentation import init_augmentations
from src.utils.common import load_json
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

    arg_parser.add_argument(
        "--val-data", type=str, default="resources/data/flores_dev.jsonl", help="Path to validation dataset."
    )
    arg_parser.add_argument(
        "--test-data", type=str, default="resources/data/flores_devtest.jsonl", help="Path to test dataset."
    )

    arg_parser.add_argument(
        "--tokenizer", type=str, default="resources/tokenizer", help="Path to tokenizer [only for vanilla model]."
    )

    arg_parser.add_argument("--vtr", action="store_true", help="Use Visual Token Representations.")

    arg_parser = VTRConfig.add_to_arg_parser(arg_parser)
    arg_parser = TransformerConfig.add_to_arg_parser(arg_parser)
    arg_parser = TrainingConfig.add_to_arg_parser(arg_parser)
    arg_parser = AugmentationConfig.add_to_arg_parser(arg_parser)
    return arg_parser


def train_langdetect_vtr(args: Namespace, val_data: list = None, test_data: list = None):
    logger.info("Training Visual Token Representation Encoder for sequence classification.")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    augmentation_config = AugmentationConfig.from_arguments(args)
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

    with open(args.probas, "rb") as f:
        probas = pickle.load(f)

    dataset_args = (char2array, vtr.window_size, vtr.stride, training_config.max_seq_len)

    nllb = NLLBDataset(probas=probas)
    lang2label = nllb.get_lang2label()
    flores_val = FloresDataset(lang2label=lang2label, split="dev")
    flores_test = FloresDataset(lang2label=lang2label, split="devtest")

    val_dataset: Dataset
    test_dataset: Dataset
    if args.no_augmentation:
        train_dataset = SlicesDataset(nllb, char2array)
        val_dataset = VTRDataset(flores_val.get_dataset(), *dataset_args)
        test_dataset = VTRDataset(flores_test.get_dataset(), *dataset_args)
    else:
        if not args.val_data and not args.test_data:
            logger.error("Need Flores noisy data files for validation and test")
            return

        with open(augmentation_config.leet) as json_file:
            leet_symbols = json.load(json_file)

        with open(augmentation_config.clusters, "rb") as f:
            cluster_symbols = pickle.load(f)

        logger.info(
            f"Noisy dataset: expected_changes_per_word:{augmentation_config.expected_changes_per_word}, proba_per_text:{augmentation_config.proba_per_text}, expected_changes_per_text:{augmentation_config.expected_changes_per_text}, max_augmentations={augmentation_config.max_augmentations}"
        )

        augmentations = init_augmentations(
            expected_changes_per_word=augmentation_config.expected_changes_per_word,
            cluster_symbols=cluster_symbols,
            leet_symbols=leet_symbols,
        )

        augmentation_dataset = AugmentationDataset(
            dataset=nllb,
            augmentations=augmentations,
            proba_per_text=augmentation_config.proba_per_text,
            expected_changes_per_text=augmentation_config.expected_changes_per_text,
            max_augmentations=augmentation_config.max_augmentations,
        )

        train_dataset = SlicesDataset(augmentation_dataset, char2array)
        val_dataset = VTRDataset(val_data, *dataset_args)
        test_dataset = VTRDataset(test_data, *dataset_args)

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


def train_langdetect_ttr(args: Namespace, val_data: list = None, test_data: list = None):
    logger.info("Training TTR for sequence classification.")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    augmentation_config = AugmentationConfig.from_arguments(args)
    with open(args.probas, "rb") as f:
        probas = pickle.load(f)
    nllb_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-3.3B")

    train_dataset: AugmentationDataset | NLLBDataset
    if args.no_augmentation:
        nllb = NLLBDataset(probas=probas, tokenizer=nllb_tokenizer, max_seq_len=training_config.max_seq_len)
        lang2label = nllb.get_lang2label()
        flores_val = FloresDataset(lang2label=lang2label, split="dev")
        flores_test = FloresDataset(lang2label=lang2label, split="devtest")
        train_dataset = nllb
        val_dataset = BERTDataset(
            flores_val.get_dataset(), tokenizer=nllb_tokenizer, max_seq_len=training_config.max_seq_len
        )
        test_dataset = BERTDataset(
            flores_test.get_dataset(), tokenizer=nllb_tokenizer, max_seq_len=training_config.max_seq_len
        )
    else:
        with open(augmentation_config.leet) as json_file:
            leet_symbols = json.load(json_file)

        with open(augmentation_config.clusters, "rb") as f:
            cluster_symbols = pickle.load(f)

        logger.info(
            f"Noisy dataset: expected_changes_per_word:{augmentation_config.expected_changes_per_word}, proba_per_text:{augmentation_config.proba_per_text}, expected_changes_per_text:{augmentation_config.expected_changes_per_text}, max_augmentations={augmentation_config.max_augmentations}"
        )

        nllb = NLLBDataset(probas=probas)

        augmentations = init_augmentations(
            expected_changes_per_word=augmentation_config.expected_changes_per_word,
            cluster_symbols=cluster_symbols,
            leet_symbols=leet_symbols,
        )

        augmentation_dataset = AugmentationDataset(
            dataset=nllb,
            augmentations=augmentations,
            proba_per_text=augmentation_config.proba_per_text,
            expected_changes_per_text=augmentation_config.expected_changes_per_text,
            max_augmentations=augmentation_config.max_augmentations,
            tokenizer=nllb_tokenizer,
            max_seq_len=training_config.max_seq_len,
        )

        train_dataset = augmentation_dataset
        val_dataset = BERTDataset(val_data, tokenizer=nllb_tokenizer, max_seq_len=training_config.max_seq_len)
        test_dataset = BERTDataset(test_data, tokenizer=nllb_tokenizer, max_seq_len=training_config.max_seq_len)

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


def main(args: Namespace):
    logger.info("Loading data...")
    val_data = load_json(args.val_data) if args.val_data else None
    test_data = load_json(args.test_data) if args.test_data else None
    if args.vtr:
        train_langdetect_vtr(args, val_data=val_data, test_data=test_data)
    else:
        train_langdetect_ttr(args, val_data=val_data, test_data=test_data)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
