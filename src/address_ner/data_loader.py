"""Build a PyTorch Dataset and dynamically padded DataLoader batches."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .tokenize_dataset import (
    DEFAULT_MODEL_NAME,
    IGNORE_INDEX,
    WordTokenizer,
    load_jsonl,
    tokenize_and_align_record,
)


# Configuration: edit these values before running this module.
DATASET_PATH = Path("data/sample.jsonl")
MODEL_NAME = DEFAULT_MODEL_NAME
MAX_LENGTH = 64
BATCH_SIZE = 4


class AddressNerDataset(Dataset[dict[str, list[int]]]):
    """Pre-tokenized, model-ready features for supervised NER training."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: WordTokenizer,
        max_length: int = 64,
    ) -> None:
        self.features = [
            tokenize_and_align_record(record, tokenizer, max_length)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        feature = self.features[index]
        return {
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"],
            "labels": feature["labels"],
        }


class DynamicPaddingCollator:
    """Pad each batch only to the longest sequence in that batch."""

    def __init__(self, pad_token_id: int, label_pad_token_id: int = IGNORE_INDEX) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")

        max_length = max(len(feature["input_ids"]) for feature in features)

        padded_input_ids: list[list[int]] = []
        padded_attention_masks: list[list[int]] = []
        padded_labels: list[list[int]] = []

        for feature in features:
            padding_length = max_length - len(feature["input_ids"])

            padded_input_ids.append(
                feature["input_ids"] + [self.pad_token_id] * padding_length
            )
            padded_attention_masks.append(
                feature["attention_mask"] + [0] * padding_length
            )
            padded_labels.append(
                feature["labels"] + [self.label_pad_token_id] * padding_length
            )

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def build_dataloader(
    dataset: AddressNerDataset,
    pad_token_id: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    collator = DynamicPaddingCollator(pad_token_id=pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    records = load_jsonl(DATASET_PATH)
    dataset = AddressNerDataset(records, tokenizer, max_length=MAX_LENGTH)
    dataloader = build_dataloader(
        dataset,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    batch = next(iter(dataloader))
    for name, tensor in batch.items():
        print(f"{name}: shape={tuple(tensor.shape)}")
        print(tensor)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
