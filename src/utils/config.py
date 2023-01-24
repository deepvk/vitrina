from argparse import Namespace, ArgumentParser
from dataclasses import dataclass, fields


@dataclass
class VTRConfig:
    font: str
    font_size: int
    window_size: int
    stride: int
    kernel_size: int
    out_channels: int
    ratio: float
    max_slices_count_per_word: int = None

    @classmethod
    def from_arguments(cls, args: Namespace) -> "VTRConfig":
        config_fields = [it.name for it in fields(cls)]
        kwargs = {it: getattr(args, it) for it in config_fields}
        return cls(**kwargs)

    @classmethod
    def add_to_arg_parser(cls, arg_parser: ArgumentParser) -> ArgumentParser:
        arg_parser.add_argument(
            "--font", type=str, default="resources/fonts/NotoSans.ttf", help="Path to font that used for rendering."
        )
        arg_parser.add_argument("--font-size", type=int, default=15, help="Font size to use for rendering.")

        arg_parser.add_argument("--window-size", type=int, default=30, help="Window size to slice the image w/ text.")
        arg_parser.add_argument("--stride", type=int, default=5, help="Window step size.")

        arg_parser.add_argument("--kernel-size", type=int, default=3, help="Kernel size to use for convolutions.")
        arg_parser.add_argument(
            "--out-channels", type=int, default=256, help="Number of output channels in the last convolutional layer."
        )

        arg_parser.add_argument("--ratio", type=float, default=0.7, help="Ratio of letter to be detected on a slice.")
        arg_parser.add_argument(
            "--max-slices-count-per-word", type=int, default=9, help="Maximum number of slices per word."
        )
        return arg_parser


@dataclass
class TransformerConfig:
    num_layers: int
    emb_size: int
    n_head: int
    dropout: float
    num_classes: int

    @classmethod
    def from_arguments(cls, args: Namespace) -> "TransformerConfig":
        config_fields = [it.name for it in fields(cls)]
        kwargs = {it: getattr(args, it) for it in config_fields}
        return cls(**kwargs)

    @classmethod
    def add_to_arg_parser(cls, arg_parser: ArgumentParser) -> ArgumentParser:
        arg_parser.add_argument("--num-layers", type=int, default=1, help="Number of layers in encoder.")
        arg_parser.add_argument("--emb-size", type=int, default=768, help="Embedding size.")
        arg_parser.add_argument("--n-head", type=int, default=12, help="Number of heads in MHA layers.")
        arg_parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
        arg_parser.add_argument("--num-classes", type=int, default=2, help="Number of labels' classes.")
        return arg_parser


@dataclass
class TrainingConfig:
    max_seq_len: int = 512
    batch_size: int = 32
    epochs: int = 10

    lr: float = 5e-5
    warmup: int = 1000
    beta1: float = 0.9
    beta2: float = 0.999

    device: str = None
    random_state: int = 21
    log_every: int = 1000
    num_workers: int = 1

    @classmethod
    def from_arguments(cls, args: Namespace) -> "TrainingConfig":
        config_fields = [it.name for it in fields(cls)]
        kwargs = {it: getattr(args, it) for it in config_fields}
        return cls(**kwargs)

    @classmethod
    def add_to_arg_parser(cls, arg_parser: ArgumentParser) -> ArgumentParser:
        arg_parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum len of tokens per sequence.")
        arg_parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
        arg_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
        arg_parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
        arg_parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps.")
        arg_parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam.")
        arg_parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Adam.")
        arg_parser.add_argument(
            "--device", type=str, default=None, help="'cuda', or 'cpu'. If 'None' it will be set automatically."
        )
        arg_parser.add_argument("--random-state", type=int, default=21, help="Random state.")
        arg_parser.add_argument("--log-every", type=int, default=1000, help="Log every N steps.")
        arg_parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for data loaders.")
        return arg_parser
