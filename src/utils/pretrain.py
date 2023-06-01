from dataclasses import asdict
import os

import torch
import wandb
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.utils.common import set_deterministic_mode, dict_to_device
from src.utils.config import TrainingConfig
from src.datasets.vtr_dataset import VTRDatasetOCR

WANDB_PROJECT_NAME = "visual-text"


def train(
    model: nn.Module,
    train_dataset: Dataset,
    config: TrainingConfig,
    *,
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
    num_training_steps = config.steps
    logger.info(f"Using AdamW optimizer | lr: {config.lr}")
    optimizer = AdamW(
        model.parameters(), config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup, num_training_steps=num_training_steps
    )
    logger.info(f"Use linear scheduler for {num_training_steps} training steps, {config.warmup} warmup steps")

    wandb.init(project=WANDB_PROJECT_NAME, config=asdict(config))
    wandb.watch(model, log="gradients", log_freq=50, idx=None, log_graph=False)

    logger.info(f"Start training for {num_training_steps} steps")

    validation_wandb_path = os.path.join(wandb.run.dir, "validation")
    if not os.path.exists(validation_wandb_path):
        os.makedirs(validation_wandb_path)

    pbar = tqdm(total=num_training_steps)
    batch_num = 0
    log_dict = {}
    need_next_iteration = True
    while need_next_iteration:
        for batch in train_dataloader:
            if batch_num == num_training_steps:
                need_next_iteration = False
                break
            model.train()

            batch_num += 1
            batch = dict_to_device(batch, except_keys={"max_word_len", "texts"}, device=device)

            optimizer.zero_grad()

            model_output = model(batch)
            loss = model_output["loss"]

            if ocr_flag:
                assert isinstance(train_dataset, VTRDatasetOCR)

                log_dict["train/CTC_loss"] = model_output["ctc_loss"]
                log_dict["train/LPIPS_loss"] = model_output["lpips_loss"]

            loss.backward()
            optimizer.step()
            scheduler.step()

            log_dict["train/loss"] = loss
            log_dict["train/learning_rate"] = scheduler.get_last_lr()[0]

            wandb.log(log_dict)

            pbar.desc = f"Step {batch_num} | Train loss: {round(loss.item(), 3)}"
            pbar.update()

            if batch_num % config.log_every == 0 and val_dataloader is not None:
                evaluate_model(
                    model,
                    val_dataloader,
                    device,
                    log=True,
                    group="val",
                    ocr_flag=ocr_flag,
                )

            if batch_num % config.save_every == 0:
                torch.save(model.state_dict(), os.path.join(validation_wandb_path, f"batch_{batch_num}.ckpt"))

    pbar.close()
    logger.info("Training finished")

    if val_dataloader is not None:
        evaluate_model(model, val_dataloader, device, log=True, group="val", ocr_flag=ocr_flag)

    if test_dataloader is not None:
        evaluate_model(model, test_dataloader, device, log=True, group="test", ocr_flag=ocr_flag)

    logger.info(f"Saving model")
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "last.ckpt"))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    log: bool = True,
    group: str = "",
    ocr_flag: bool = False,
):
    if log:
        logger.info(f"Evaluating the model on {group} set")

    model.eval()
    loss = 0
    ctc_loss = 0
    lpips_loss = 0
    for test_batch in tqdm(dataloader, leave=False, position=0):
        batch = dict_to_device(test_batch, except_keys={"max_word_len", "texts"}, device=device)
        output = model(batch)

        loss += output["loss"]
        if ocr_flag:
            ctc_loss += output["ctc_loss"]
            lpips_loss += output["lpips_loss"]

    losses_dict = {f"{group}/loss": loss / len(dataloader)}
    if ocr_flag:
        losses_dict[f"{group}/LPIPS_loss"] = lpips_loss / len(dataloader)
        losses_dict[f"{group}/CTC_loss"] = ctc_loss / len(dataloader)

    if log:
        wandb.log(losses_dict)
    logger.info(",\n ".join(f"{k}: {v}" for k, v in losses_dict.items()))
