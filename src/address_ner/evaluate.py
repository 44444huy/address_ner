"""Evaluate exact-match entity Precision, Recall and F1."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .inference import predict_record
from .labels import ENTITY_TYPES
from .tokenize_dataset import load_jsonl


# Configuration: edit these values before running this module.
DATASET_PATH = Path("data/synthetic/processed-v2-stratified/test.jsonl")
CHECKPOINT_PATH = Path("checkpoints/synthetic-v3")
MAX_LENGTH = 64


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def _entity_key(entity: dict[str, Any]) -> tuple[str, int, int]:
    return entity["type"], entity["start_word"], entity["end_word"]


def _scores(counts: dict[str, int]) -> dict[str, int | float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        **counts,
        "support": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_entity_metrics(
    prediction_results: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compute micro and per-type exact-span entity metrics."""
    overall = _empty_counts()
    per_type = {entity_type: _empty_counts() for entity_type in sorted(ENTITY_TYPES)}

    for result in prediction_results:
        predicted = {
            _entity_key(entity) for entity in result["predicted_entities"]
        }
        gold = {_entity_key(entity) for entity in result["gold_entities"]}

        true_positives = predicted & gold
        false_positives = predicted - gold
        false_negatives = gold - predicted

        overall["tp"] += len(true_positives)
        overall["fp"] += len(false_positives)
        overall["fn"] += len(false_negatives)

        for entity_type, _, _ in true_positives:
            per_type[entity_type]["tp"] += 1
        for entity_type, _, _ in false_positives:
            per_type[entity_type]["fp"] += 1
        for entity_type, _, _ in false_negatives:
            per_type[entity_type]["fn"] += 1

    return {
        "overall": _scores(overall),
        "per_type": {
            entity_type: _scores(counts)
            for entity_type, counts in per_type.items()
        },
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT_PATH,
        local_files_only=True,
    ).to(device)
    model.eval()

    records = load_jsonl(DATASET_PATH)
    prediction_results = [
        predict_record(model, tokenizer, record, device, MAX_LENGTH)
        for record in records
    ]
    report = {
        "dataset": str(DATASET_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(device),
        "num_records": len(records),
        **compute_entity_metrics(prediction_results),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
