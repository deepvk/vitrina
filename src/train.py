import argparse
from typing import Callable, Tuple, List, Dict, Union, Optional

import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.datasets.bert_dataset import BERTDataset
from src.datasets.bert_dataset_sl import BERTDatasetSL
from src.datasets.sized_collated_dataset import SizedCollatedDataset
from src.datasets.vtr_dataset import VTRDataset
from src.datasets.vtr_dataset_sl import VTRDatasetSL
from src.models.transformer_encoder.encoder import Encoder
from src.models.transformer_encoder.encoder_for_sequence_labeling import (
    EncoderForSequenceLabeling,
)
from src.models.vtr.classifier import VisualToxicClassifier
from src.models.vtr.sequence_labeler import VisualTextSequenceLabeler
from src.utils.utils import dict_to_device, load_json, set_deterministic_mode


def split_dataset(dataset: SizedCollatedDataset, test_size: float, random_state: int) -> Tuple[Dataset, Dataset]:
    dataset_size = len(dataset)
    test_dataset_size = int(test_size * dataset_size)
    train_dataset_size = dataset_size - test_dataset_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_dataset_size, test_dataset_size],
        generator=torch.Generator().manual_seed(random_state),
    )
    return train_dataset, test_dataset


def train(
    train_data: str = "resources/data/train_dataset.jsonl",
    val_data: Optional[str] = "resources/data/val_dataset.jsonl",
    test_data: Optional[str] = "resources/data/test_dataset.jsonl",
    tokenizer: Optional[str] = None,
    max_seq_len: int = 512,
    max_slices_count_per_word: int = 9,
    font_size: int = 15,
    window_size: int = 30,
    stride: int = 5,
    epochs: int = 10,
    random_state: int = 21,
    log_every: int = 1000,
    emb_size: int = 768,
    num_workers: int = 1,
    save_to: Optional[str] = None,
    font: str = "resources/fonts/NotoSans.ttf",
    out_channels: int = 256,
    nhead: int = 12,
    beta1: float = 0.9,
    beta2: float = 0.999,
    sl: bool = False,
    dropout: float = 0.0,
    device: Optional[str] = None,
    warmup: int = 1000,
    batch_size: int = 32,
    kernel_size: int = 3,
    num_layers: int = 1,
    lr: float = 5e-5,
):
    """
    Trains model

    :param train_data: path to training data (default: resources/data/train_dataset.jsonl)
    :param val_data: path to validation data (default: resources/data/val_dataset.jsonl)
    :param test_data: path to test data (default: resources/data/test_dataset.jsonl)
    :param tokenizer: path to pretrained tokenizer directory. If None, vtr model will be trained (default: None)
    :param max_seq_len: used to create datasets. In the case of vtr, a limit on the number of slices, in the case of a classic transformer, on the number of tokens. (default: 512)
    :param max_slices_count_per_word: used to limit the number of slices for one word. You can learn more about the process of processing a sequence of words by looking at the image `resources/images/padding_in_sl.jpg` (default: 9)
    :param font_size: font size with which the image is generated (default: 15)
    :param window_size: window width (default: 30)
    :param stride: window step size (default: 5)
    :param epochs: number of training epochs (default: 10)
    :param random_state: (default: 21)
    :param log_every: how often to recalculate the value of metrics on the validation set. These values are logged to wandb (default: 1000)
    :param emb_size: embedding size received by visual representations (default: 768)
    :param num_workers: number of threads in DataLoader (default: 1)
    :param save_to: the path where the weights of the model are saved after training, if a value is specified, if not, then the weights are not saved (default: None)
    :param font: font used to generate character images (default: resources/fonts/NotoSans.ttf)
    :param out_channels: number of output channels in the last convolutional layer (default: 256)
    :param nhead: number of transformer layer heads (default: 12)
    :param beta1: Adam optimizer parameter (default: 0.9)
    :param beta2: Adam optimizer parameter (default: 0.999)
    :param sl: if true, then the task of labeling the sequence is being solved (default: False)
    :param dropout: dropout probability (default: 0.0)
    :param device: 'cuda' or 'cpu'. If None it will be set automatically (default: None)
    :param warmup: number of steps to warm up Linear Scheduler (default: 1000)
    :param batch_size: batch size (default: 32)
    :param kernel_size: convolution kernel size (default 3)
    :param num_layers: number of transformer encoder layers in the visual representation model (default: 1)
    :param lr: learning rate (default: 5e-5)
    :return:
    """
    wandb.init(project="visual-text", entity="borisshapa")
    set_deterministic_mode(random_state)

    actual_device: str
    if device is not None:
        actual_device = device
    else:
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"

    labeled_texts = load_json(train_data)

    val_labeled_texts = None
    if val_data is not None:
        val_labeled_texts = load_json(val_data)

    test_labeled_texts = None
    if test_data is not None:
        test_labeled_texts = load_json(test_data)

    model: nn.Module
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    dataset: SizedCollatedDataset
    val_dataset: Optional[SizedCollatedDataset] = None
    test_dataset: Optional[SizedCollatedDataset] = None

    if sl:
        criterion = BceLossForTokenClassification()
        create_dataset_sl_f: Callable[
            [List[Dict[str, Union[List[List[Union[str, int]]], int]]]],
            SizedCollatedDataset,
        ]
        if not tokenizer:
            model = VisualTextSequenceLabeler(
                height=font_size,
                width=window_size,
                kernel_size=kernel_size,
                emb_size=emb_size,
                num_layers=num_layers,
                out_channels=out_channels,
                nhead=nhead,
                dropout=dropout,
            )
            create_dataset_sl_f = lambda texts: VTRDatasetSL(
                labeled_texts=texts,
                font=font,
                font_size=font_size,
                window_size=window_size,
                stride=stride,
                max_seq_len=max_seq_len,
                max_slices_count_per_word=max_slices_count_per_word,
            )
        else:
            model = EncoderForSequenceLabeling(dropout=dropout, num_layers=num_layers)
            tokenizer_path: str = tokenizer
            create_dataset_sl_f = lambda texts: BERTDatasetSL(
                texts,
                tokenizer_path,
                max_seq_len,
            )
        train_dataset = create_dataset_sl_f(labeled_texts)
        if val_labeled_texts is not None:
            val_dataset = create_dataset_sl_f(val_labeled_texts)
        if test_labeled_texts is not None:
            test_dataset = create_dataset_sl_f(test_labeled_texts)
    else:
        criterion = nn.BCEWithLogitsLoss()
        create_dataset_f: Callable[
            [List[Dict[str, Union[str, int]]]],
            SizedCollatedDataset,
        ]
        if not tokenizer:
            model = VisualToxicClassifier(
                height=font_size,
                width=window_size,
                kernel_size=kernel_size,
                emb_size=emb_size,
                num_layers=num_layers,
                nhead=nhead,
                out_channels=out_channels,
                dropout=dropout,
            )
            create_dataset_f = lambda texts: VTRDataset(
                texts,
                font,
                font_size,
                window_size,
                stride,
                max_seq_len,
            )
        else:
            model = Encoder(dropout=dropout, num_layers=num_layers)
            tokenizer_path = tokenizer
            create_dataset_f = lambda texts: BERTDataset(texts, tokenizer_path, max_seq_len)
        train_dataset = create_dataset_f(labeled_texts)
        if val_labeled_texts:
            val_dataset = create_dataset_f(val_labeled_texts)
        if test_labeled_texts:
            test_dataset = create_dataset_f(test_labeled_texts)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_function,
        num_workers=num_workers,
        shuffle=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        actual_val_dataset: SizedCollatedDataset = val_dataset
        val_dataloader = DataLoader(
            actual_val_dataset,
            batch_size=batch_size,
            collate_fn=actual_val_dataset.collate_function,
            num_workers=num_workers,
            shuffle=False,
        )

    test_dataloader = None
    if test_dataset is not None:
        actual_test_dataset: SizedCollatedDataset = test_dataset
        test_dataloader = DataLoader(
            actual_test_dataset,
            batch_size=batch_size,
            collate_fn=actual_test_dataset.collate_function,
            num_workers=num_workers,
            shuffle=False,
        )

    parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("COUNT: ", parameters_count)
    model.to(actual_device)

    optimizer = AdamW(model.parameters(), lr, betas=(beta1, beta2))

    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=num_training_steps
    )

    pbar = tqdm(total=num_training_steps)

    batch_num = 0

    wandb.watch(model, criterion, log="gradients", log_freq=50, idx=None, log_graph=(False))

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_dataloader:
            batch_num += 1

            optimizer.zero_grad()

            batch = dict_to_device(
                batch,
                except_keys={"max_word_len"},
                device=actual_device,
            )
            prediction = model(batch)

            loss = criterion(prediction, batch["labels"])

            loss.backward()

            optimizer.step()
            scheduler.step()

            wandb.log({"train loss": loss, "learning rate": scheduler.get_last_lr()[0]})
            pbar.desc = f"Epoch {epoch} / {epochs} | Train loss: {loss}"
            pbar.update()

            if batch_num % log_every == 0 and val_dataloader is not None:
                actual_val_dataloader: DataLoader = val_dataloader
                model.eval()

                with torch.no_grad():
                    ground_truth = []
                    predictions = []

                    for test_batch in actual_val_dataloader:
                        output = model(
                            dict_to_device(
                                test_batch,
                                except_keys={"max_word_len"},
                                device=actual_device,
                            )
                        )
                        true_labels = test_batch["labels"]

                        if sl:
                            true_labels = true_labels.view(-1)
                            mask = true_labels >= 0

                            true_labels = true_labels[mask]
                            output = output.view(-1)[mask]

                        predictions.append((output > 0).to(torch.float).cpu().detach())
                        ground_truth.append(true_labels.cpu().detach())

                    ground_truth = torch.cat(ground_truth).numpy()
                    predictions = torch.cat(predictions).numpy()

                    wandb.log({"accuracy": accuracy_score(ground_truth, predictions)})
                    wandb.log({"precision": precision_score(ground_truth, predictions, zero_division=0)})
                    wandb.log({"recall": recall_score(ground_truth, predictions, zero_division=0)})
                    wandb.log({"f1": f1_score(ground_truth, predictions, pos_label=1, zero_division=0)})

                model.train()

    pbar.close()

    if test_dataloader is not None:
        # TODO: logic to evaluate model on test dataset
        pass

    if save_to is not None:
        torch.save(model.state_dict(), save_to)


class BceLossForTokenClassification:
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = outputs.view(-1)
        labels = labels.view(-1).float()
        mask = (labels >= 0).float()
        num_tokens = int(torch.sum(mask))
        loss = self.bce_loss(outputs, labels) * mask
        return torch.sum(loss) / num_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=f"resources/data/train_dataset.jsonl",
    )
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-slices-count-per-word", type=int, default=9)
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=21)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--emb-size", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--save-to", type=str, default=None)
    parser.add_argument("--font", type=str, default="resources/fonts/NotoSans.ttf")
    parser.add_argument("--out-channels", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--sl", action="store_true")

    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()

    train(**vars(args))
