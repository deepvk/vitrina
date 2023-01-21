from dataclasses import asdict
from os.path import join

import torch
import wandb
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.nn import CTCLoss
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from src.utils.common import set_deterministic_mode, dict_to_device
from src.utils.config import TrainingConfig
from src.utils.common import char2int


WANDB_PROJECT_NAME = "visual-text"


def compute_ctc_loss(criterion, logits, texts):
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    input_lengths = torch.LongTensor([log_probs.shape[0]] * log_probs.shape[1])

    chars = list("".join(np.concatenate(texts).flatten()))
    targets = char2int(chars)

    get_len = np.vectorize(len)
    target_lengths = pad_sequence([torch.from_numpy(get_len(arr)) for arr in texts], batch_first=True, padding_value=0)

    ctc_loss = criterion(log_probs, targets, input_lengths, target_lengths)
    ctc_loss /= input_lengths.shape[0]

    return ctc_loss


def train(
    model: nn.Module,
    train_dataset: Dataset,
    criterion: nn.Module,
    config: TrainingConfig,
    *,
    sl: bool,
    val_dataset: Dataset = None,
    test_dataset: Dataset = None,
    ocr_flag: bool = False,
):
    logger.info(f"Fix random state: {config.random_state}")
    set_deterministic_mode(config.random_state)

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Create train dataloader | batch size: {config.batch_size}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=train_dataset.collate_function,  # type: ignore
        num_workers=config.num_workers,
        shuffle=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        logger.info(f"Create val dataloader")
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=val_dataset.collate_function,  # type: ignore
            num_workers=config.num_workers,
            shuffle=False,
        )

    test_dataloader = None
    if test_dataset is not None:
        logger.info(f"Create test dataloader")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            collate_fn=test_dataset.collate_function,  # type: ignore
            num_workers=config.num_workers,
            shuffle=False,
        )

    parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters count: {parameters_count}")
    model.to(device)

    ctc_criterion = CTCLoss(reduction="sum", zero_infinity=True)

    logger.info(f"Using AdamW optimizer | lr: {config.lr}")
    optimizer = AdamW(model.parameters(), config.lr, betas=(config.beta1, config.beta2))
    num_training_steps = len(train_dataloader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup, num_training_steps=num_training_steps
    )
    logger.info(f"Use linear scheduler for {num_training_steps} training steps, {config.warmup} warmup steps")

    wandb.init(project=WANDB_PROJECT_NAME, config=asdict(config))
    wandb.watch(model, criterion, log="gradients", log_freq=50, idx=None, log_graph=False)

    logger.info(f"Start training for {config.epochs} epochs")
    pbar = tqdm(total=num_training_steps)
    batch_num = 0
    log_dict = {}
    for epoch in range(1, config.epochs + 1):
        for batch in train_dataloader:
            model.train()

            batch_num += 1
            batch = dict_to_device(batch, except_keys={"max_word_len", "texts"}, device=device)

            optimizer.zero_grad()

            model_output = model(batch)
            loss = criterion(model_output["logits"], batch["labels"])

            if ocr_flag:
                ctc_loss = compute_ctc_loss(ctc_criterion, model_output["ocr logits"], batch["texts"])

                log_dict["train/ctc_loss"] = ctc_loss
                log_dict["train/bce_loss"] = loss

                loss += 0.4 * ctc_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            log_dict["train/loss"] = loss
            log_dict["train/learning_rate"] = scheduler.get_last_lr()[0]

            wandb.log(log_dict)

            pbar.desc = f"Epoch {epoch} / {config.epochs} | Train loss: {round(loss.item(), 3)}"
            pbar.update()

            if batch_num % config.log_every == 0 and val_dataloader is not None:
                evaluate_model(model, val_dataloader, device, sl, log=True, group="val")
    pbar.close()
    logger.info("Training finished")

    if val_dataloader is not None:
        evaluate_model(model, val_dataloader, device, sl, log=True, group="val")

    if test_dataloader is not None:
        evaluate_model(model, test_dataloader, device, sl, log=True, group="test")

    logger.info(f"Saving model")
    torch.save(model.state_dict(), join(wandb.run.dir, "last.ckpt"))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    sl: bool,
    *,
    log: bool = True,
    group: str = "",
) -> dict[str, float]:
    if log:
        logger.info(f"Evaluating the model on {group} set")

    model.eval()
    ground_truth = []
    predictions = []
    for test_batch in tqdm(dataloader, leave=False):
        batch = dict_to_device(test_batch, except_keys={"max_word_len", "texts"}, device=device)

        output = model(batch)["logits"]

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

    precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth, predictions, average="binary")
    accuracy = accuracy_score(ground_truth, predictions)
    result = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1_score}

    if group != "":
        result = {f"{group}/{k}": v for k, v in result.items()}

    if log:
        wandb.log(result)
        log_string = ", ".join(f"{k}: {round(v, 3)}" for k, v in result.items())
        logger.info(log_string)

    return result
