"""Evaluate a checkpoint on fixed character-span golden cases."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .convert_annotations import VnCoreNlpWordSegmenter, convert_record
from .evaluate import compute_entity_metrics
from .golden_cases import validate_golden_jsonl
from .inference import predict_record
from .validate_annotations import load_annotation_jsonl


# Configuration: edit these values before running this module.
DATASET_PATH = Path("data/golden/golden_cases.seed.jsonl")
CHECKPOINT_PATH = Path("checkpoints/synthetic-v3")
MODEL_DIR = Path("tools/vncorenlp")
MAX_LENGTH = 64
OUTPUT_PATH: Path | None = Path("reports/golden.synthetic-v3.json")


def _entity_signature(entity: dict[str, Any]) -> tuple[str, int, int]:
    return entity["type"], entity["start_word"], entity["end_word"]


def _is_exact_case(result: dict[str, Any]) -> bool:
    predicted = {
        _entity_signature(entity) for entity in result["predicted_entities"]
    }
    gold = {_entity_signature(entity) for entity in result["gold_entities"]}
    return predicted == gold


def _exact_case_scores(results: list[dict[str, Any]]) -> dict[str, int | float]:
    num_exact_cases = sum(_is_exact_case(result) for result in results)
    return {
        "num_cases": len(results),
        "num_exact_cases": num_exact_cases,
        "exact_case_accuracy": num_exact_cases / len(results) if results else 0.0,
    }


def build_golden_report(
    prediction_results: list[dict[str, Any]],
    categories_by_id: dict[str, list[str]],
) -> dict[str, Any]:
    """Build overall metrics, category slices and compact failure details."""
    results_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failed_cases: list[dict[str, Any]] = []

    for result in prediction_results:
        categories = categories_by_id[result["id"]]
        for category in categories:
            results_by_category[category].append(result)

        if not _is_exact_case(result):
            failed_cases.append(
                {
                    "id": result["id"],
                    "raw_text": result["raw_text"],
                    "categories": categories,
                    "expected": [
                        {"type": entity["type"], "text": entity["text"]}
                        for entity in result["gold_entities"]
                    ],
                    "predicted": [
                        {
                            "type": entity["type"],
                            "text": entity["text"],
                            "mean_confidence": entity["mean_confidence"],
                        }
                        for entity in result["predicted_entities"]
                    ],
                }
            )

    return {
        **_exact_case_scores(prediction_results),
        "entity_metrics": compute_entity_metrics(prediction_results),
        "per_category": {
            category: {
                **_exact_case_scores(results),
                "entity_metrics": compute_entity_metrics(results),
            }
            for category, results in sorted(results_by_category.items())
        },
        "failed_cases": failed_cases,
    }


def main() -> int:
    dataset_path = DATASET_PATH.resolve()
    checkpoint_path = CHECKPOINT_PATH.resolve()
    model_dir = MODEL_DIR.resolve()
    _, _, issues = validate_golden_jsonl(dataset_path)
    if issues:
        print(f"FAILED: golden dataset has {len(issues)} issue(s)")
        for issue in issues:
            print(f"- {issue}")
        return 1

    golden_records = load_annotation_jsonl(dataset_path)
    segmenter = VnCoreNlpWordSegmenter(model_dir)
    bio_records = [convert_record(record, segmenter) for record in golden_records]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_path, local_files_only=True
    ).to(device)
    model.eval()
    prediction_results = [
        predict_record(model, tokenizer, record, device, MAX_LENGTH)
        for record in bio_records
    ]
    report = {
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        **build_golden_report(
            prediction_results,
            {record["id"]: record["categories"] for record in golden_records},
        ),
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if OUTPUT_PATH:
        output_path = OUTPUT_PATH.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote golden evaluation report to {output_path}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
