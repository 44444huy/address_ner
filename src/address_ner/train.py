"""Minimal full-fine-tuning loop for the PhoBERT address NER baseline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from .data_loader import AddressNerDataset, build_dataloader
from .forward_pass import load_model
from .tokenize_dataset import DEFAULT_MODEL_NAME, IGNORE_INDEX, load_jsonl


# Configuration: edit these values before starting training.
TRAIN_FILE = Path("data/synthetic/processed-v2-stratified/train.jsonl")
VALIDATION_FILE = Path("data/synthetic/processed-v2-stratified/validation.jsonl")
OUTPUT_DIR = Path("checkpoints/synthetic-v3")
MODEL_NAME = DEFAULT_MODEL_NAME
BATCH_SIZE = 4
MAX_LENGTH = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def supervised_token_count(labels: torch.Tensor) -> int:
    return int((labels != IGNORE_INDEX).sum().item())


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    model.train()
    weighted_loss = 0.0
    token_count = 0

    for batch in dataloader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

        current_tokens = supervised_token_count(batch["labels"])
        weighted_loss += outputs.loss.item() * current_tokens
        token_count += current_tokens

    return weighted_loss / token_count


def evaluate_loss(model: torch.nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    weighted_loss = 0.0
    token_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch(batch, device)
            outputs = model(**batch)

            current_tokens = supervised_token_count(batch["labels"])
            weighted_loss += outputs.loss.item() * current_tokens
            token_count += current_tokens

    return weighted_loss / token_count


def main() -> int:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(MODEL_NAME).to(device)

    train_dataset = AddressNerDataset(
        load_jsonl(TRAIN_FILE), tokenizer, max_length=MAX_LENGTH
    )
    validation_dataset = AddressNerDataset(
        load_jsonl(VALIDATION_FILE), tokenizer, max_length=MAX_LENGTH
    )

    train_loader = build_dataloader(
        train_dataset,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_loss = float("inf")
    history: list[dict[str, float | int]] = []

    print(f"device={device}, train={len(train_dataset)}, validation={len(validation_dataset)}")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_loss = evaluate_loss(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch={epoch}/{EPOCHS} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"saved best checkpoint to {OUTPUT_DIR}")

    summary = {
        "model_name": MODEL_NAME,
        "device": str(device),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": RANDOM_SEED,
        "best_validation_loss": best_validation_loss,
        "history": history,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
