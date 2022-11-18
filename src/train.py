import argparse
import json
from typing import Callable, Tuple

import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from datasets.bert_dataset import BERTDataset
from datasets.bert_dataset_sl import BERTDatasetSL
from datasets.vtr_dataset import VTRDataset
from datasets.vtr_dataset_sl import VTRDatasetSL
from models.transformer_encoder.encoder import Encoder
from models.transformer_encoder.encoder_for_sequence_labeling import (
    EncoderForSequenceLabeling,
)
from models.vtr.classifier import VisualToxicClassifier
from models.vtr.sequence_labeler import VisualTextSequenceLabeler
from utils.utils import dict_to_device


def split_dataset(
    dataset: Dataset, test_size: float, random_state: int
) -> Tuple[Dataset, Dataset]:
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
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    epochs: int,
    test_size: float,
    val_size: float,
    random_state: int,
    log_every: int,
    lr: float,
    num_workers: int,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    warmup: int,
    beta1: float,
    beta2: float,
    sl: bool = False,
    test_dataset: Dataset = None,
    save_to: str = None,
    val: bool = False,
):
    if test_dataset is not None:
        train_dataset = dataset
    else:
        train_dataset, test_dataset = split_dataset(dataset, test_size, random_state)
        train_dataset, val_dataset = split_dataset(
            dataset, val_size / (1 - test_size), random_state
        )

        if val:
            test_dataset = val_dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_function,
        num_workers=num_workers,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_function,
        num_workers=num_workers,
        shuffle=False,
    )

    parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("COUNT: ", parameters_count)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr, betas=(beta1, beta2))

    num_training_steps = ((len(train_dataset) + batch_size - 1) // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=num_training_steps
    )

    pbar = tqdm(total=num_training_steps)

    batch_num = 0

    wandb.watch(
        model, criterion, log="gradients", log_freq=50, idx=None, log_graph=(False)
    )

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_dataloader:
            batch_num += 1

            optimizer.zero_grad()

            batch = dict_to_device(
                batch,
                except_keys={"max_word_len"},
                device=device,
            )
            prediction = model(batch)

            loss = criterion(prediction, batch["labels"])

            loss.backward()

            optimizer.step()
            scheduler.step()

            wandb.log({"train loss": loss, "learning rate": scheduler.get_lr()[0]})
            pbar.desc = f"Epoch {epoch} / {epochs} | Train loss: {loss}"
            pbar.update()

            if batch_num % log_every == 0:
                model.eval()

                with torch.no_grad():
                    ground_truth = []
                    predictions = []

                    for test_batch in test_dataloader:
                        output = model(
                            dict_to_device(
                                test_batch, except_keys={"max_word_len"}, device=device
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
                    wandb.log(
                        {
                            "precision": precision_score(
                                ground_truth, predictions, zero_division=0
                            )
                        }
                    )
                    wandb.log(
                        {
                            "recall": recall_score(
                                ground_truth, predictions, zero_division=0
                            )
                        }
                    )
                    wandb.log(
                        {
                            "f1": f1_score(
                                ground_truth, predictions, pos_label=1, zero_division=0
                            )
                        }
                    )

                model.train()

    pbar.close()
    if save_to is not None:
        torch.save(model.state_dict(), save_to)


class BceLossForTokenClassification:
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        outputs = outputs.view(-1)
        labels = labels.view(-1).float()
        mask = (labels >= 0).float()
        num_tokens = int(torch.sum(mask))
        loss = self.bce_loss(outputs, labels) * mask
        return torch.sum(loss) / num_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    num = 8
    parser.add_argument(
        "--data",
        type=str,
        default=f"resources/data/noisy_dataset.jsonl",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=f"tokenizer"
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-slices-count-per-word", type=int, default=9)
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=21)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--emb-size", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--save-to", type=str, default=None)
    parser.add_argument("--font", type=str, default="fonts/NotoSans.ttf")
    parser.add_argument("--out-channels", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--sl", action="store_true")
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--val", action="store_true")

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()

    wandb.init(project="visual-text", entity="borisshapa", config=vars(args))

    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = None
    model = None
    criterion = None

    create_dataset_f = None
    if args.sl:
        criterion = BceLossForTokenClassification()
        if not args.tokenizer:
            model = VisualTextSequenceLabeler(
                height=args.font_size,
                width=args.window_size,
                kernel_size=args.kernel_size,
                emb_size=args.emb_size,
                num_layers=args.num_layers,
                out_channels=args.out_channels,
                nhead=args.nhead,
                dropout=args.dropout,
            )
            create_dataset_f = lambda texts: VTRDatasetSL(
                labeled_texts=texts,
                font=args.font,
                font_size=args.font_size,
                window_size=args.window_size,
                stride=args.stride,
                max_seq_len=args.max_seq_len,
                max_slices_count_per_word=args.max_slices_count_per_word,
            )
        else:
            model = EncoderForSequenceLabeling()
            create_dataset_f = lambda texts: BERTDatasetSL(
                labeled_texts,
                args.tokenizer,
                args.max_seq_len,
            )
    else:
        criterion = nn.BCEWithLogitsLoss()
        if not args.tokenizer:
            model = VisualToxicClassifier(
                height=args.font_size,
                width=args.window_size,
                kernel_size=args.kernel_size,
                emb_size=args.emb_size,
                num_layers=args.num_layers,
                nhead=args.nhead,
                out_channels=args.out_channels,
                dropout=args.dropout,
            )
            create_dataset_f = lambda texts: VTRDataset(
                texts,
                args.font,
                args.font_size,
                args.window_size,
                args.stride,
                args.max_seq_len,
            )
        else:
            model = Encoder(args.dropout)
            create_dataset_f = lambda texts: BERTDataset(
                texts, args.tokenizer, args.max_seq_len
            )

    with open(args.data) as train_file:
        json_list = list(train_file)
    labeled_texts = list(map(json.loads, json_list))

    dataset = create_dataset_f(labeled_texts)

    test_dataset = None
    if args.test_data is not None:
        with open(args.test_data) as test_file:
            json_list = list(test_file)
        labeled_texts = list(map(json.loads, json_list))
        test_dataset = create_dataset_f(labeled_texts)

    train(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        log_every=args.log_every,
        lr=args.lr,
        num_workers=args.num_workers,
        criterion=criterion,
        save_to=args.save_to,
        sl=args.sl,
        test_dataset=test_dataset,
        warmup=args.warmup,
        beta1=args.beta1,
        beta2=args.beta2,
        val=args.val,
    )
