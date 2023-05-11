import pickle
from argparse import ArgumentParser, Namespace

from loguru import logger
from src.datasets.translation_datasets import NLLBDataset, FloresDataset
from src.datasets.common import AugmentationDataset, SlicesDataset
from src.datasets.vtr_dataset import VTRDataset
from src.models.embedders.vtr import VTREmbedder
from src.models.tasks import SequenceClassifier
from src.utils.common import load_json
from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig
from src.utils.train import train
from torch.utils.data import Dataset, IterableDataset


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--val-data", type=str, default=None, help="Path to val dataset.")
    arg_parser.add_argument("--test-data", type=str, default=None, help="Path to test dataset.")

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

    arg_parser = VTRConfig.add_to_arg_parser(arg_parser)
    arg_parser = TransformerConfig.add_to_arg_parser(arg_parser)
    arg_parser = TrainingConfig.add_to_arg_parser(arg_parser)
    return arg_parser


def train_langdetect(args: Namespace):
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

    with open(args.probas, "rb") as f:
        probas = pickle.load(f)

    dataset_args = (char2array, vtr.window_size, vtr.stride, training_config.max_seq_len)

    nllb = NLLBDataset(probas=probas)
    lang2label = nllb.get_lang2label()
    flores_val = FloresDataset(lang2label=lang2label, split="dev")
    flores_test = FloresDataset(lang2label=lang2label, split="devtest")
    train_dataset = SlicesDataset(nllb, char2array)
    val_dataset: Dataset = VTRDataset(flores_val.get_dataset(), *dataset_args)
    test_dataset: Dataset = VTRDataset(flores_test.get_dataset(), *dataset_args)

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
    train_langdetect(args)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
