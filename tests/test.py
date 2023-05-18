import json
import pickle
from argparse import ArgumentParser, Namespace

from loguru import logger
from transformers import NllbTokenizer

from src.datasets.common import AugmentationDataset, SlicesDataset, TokenizedDataset
from src.datasets.translation_datasets import NLLBDataset, FloresDataset
from src.datasets.vtr_dataset import VTRDataset
from src.models.embedders.ttr import TTREmbedder
from src.models.embedders.vtr import VTREmbedder
from src.models.tasks import SequenceClassifier
from src.utils.augmentation import init_augmentations
from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig, AugmentationConfig
from src.utils.train import train

import torch

with open("resources/nllb/probas_3.pkl", "rb") as f:
    probas = pickle.load(f)

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
        "--dataset-dir", type=str, help="Directory of validation and test Flores files."
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

args = configure_arg_parser().parse_args()

with open(args.char2array, "rb") as f:
    char2array = pickle.load(f)
  
vtr = VTRConfig.from_arguments(args)
channels = (1, 64, 128, vtr.out_channels)

model_config = TransformerConfig.from_arguments(args)
training_config = TrainingConfig.from_arguments(args)

train_dataset = NLLBDataset(probas=probas)

lang2label = train_dataset.get_lang2label()

train_dataset = SlicesDataset(train_dataset, char2array)

embedder = VTREmbedder(
            height=vtr.font_size,
            width=vtr.window_size,
            conv_kernel_size=vtr.conv_kernel_size,
            pool_kernel_size=vtr.pool_kernel_size,
            emb_size=model_config.emb_size,
            channels=channels,
        )

model_config.num_classes = train_dataset.get_num_classes()
model = SequenceClassifier(config=model_config, embedder=embedder, max_position_embeddings=training_config.max_seq_len)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
model.to(device)
print(1)
