"""Run one supervised PhoBERT token-classification forward pass."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .data_loader import AddressNerDataset, build_dataloader
from .labels import ID2LABEL, LABEL2ID, LABELS
from .tokenize_dataset import DEFAULT_MODEL_NAME, load_jsonl


DATASET_PATH = Path("data/sample.jsonl")
MODEL_NAME = DEFAULT_MODEL_NAME
MAX_LENGTH = 64
BATCH_SIZE = 4


def load_model(model_name: str) -> AutoModelForTokenClassification:
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(MODEL_NAME).to(device)

    records = load_jsonl(DATASET_PATH)
    dataset = AddressNerDataset(records, tokenizer, max_length=MAX_LENGTH)
    dataloader = build_dataloader(
        dataset,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    batch = next(iter(dataloader))
    batch = {name: tensor.to(device) for name, tensor in batch.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**batch)

    predictions = outputs.logits.argmax(dim=-1)

    print(f"device: {device}")
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"logits shape: {tuple(outputs.logits.shape)}")
    print(f"predictions shape: {tuple(predictions.shape)}")
    print(f"loss: {outputs.loss.item():.6f}")
    print("Note: the classification head is new, so predictions are random before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
