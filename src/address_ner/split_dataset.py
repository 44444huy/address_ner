"""Split JSONL records by group_id to prevent variant leakage."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .tokenize_dataset import load_jsonl


# Configuration: edit these values before running this module.
DATASET_PATH = Path("data/synthetic/bio-v2.jsonl")
OUTPUT_DIR = Path("data/synthetic/processed-v2-stratified")
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42
STRATIFY_FIELD: str | None = "generation.kind"


def split_by_group(
    records: list[dict[str, Any]],
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_field: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_ratio <= 0 or test_ratio <= 0:
        raise ValueError("validation_ratio and test_ratio must be positive")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)

    if len(groups) < 3:
        raise ValueError("At least three groups are required for train/validation/test")

    strata: dict[str, list[str]] = defaultdict(list)
    if stratify_field is None:
        strata["all"] = sorted(groups)
    else:
        for group_id, group_records in groups.items():
            values = {
                str(_nested_value(record, stratify_field, "__missing__"))
                for record in group_records
            }
            if len(values) != 1:
                raise ValueError(
                    f"group {group_id!r} has multiple values for {stratify_field}: {sorted(values)}"
                )
            strata[next(iter(values))].append(group_id)

    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    test_ids: set[str] = set()
    for stratum, stratum_group_ids in sorted(strata.items()):
        if len(stratum_group_ids) < 3:
            raise ValueError(
                f"stratum {stratum!r} needs at least three groups, found {len(stratum_group_ids)}"
            )
        current_train, current_validation, current_test = _split_group_ids(
            sorted(stratum_group_ids),
            validation_ratio,
            test_ratio,
            f"{seed}:{stratum}",
        )
        train_ids.update(current_train)
        validation_ids.update(current_validation)
        test_ids.update(current_test)

    group_order = sorted(groups)

    def collect(selected_ids: set[str]) -> list[dict[str, Any]]:
        return [
            record
            for group_id in group_order
            if group_id in selected_ids
            for record in groups[group_id]
        ]

    return collect(train_ids), collect(validation_ids), collect(test_ids)


def _nested_value(record: dict[str, Any], field: str, default: Any) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _split_group_ids(
    group_ids: list[str],
    validation_ratio: float,
    test_ratio: float,
    seed: str,
) -> tuple[set[str], set[str], set[str]]:
    random.Random(seed).shuffle(group_ids)
    validation_count = max(1, round(len(group_ids) * validation_ratio))
    test_count = max(1, round(len(group_ids) * test_ratio))

    while validation_count + test_count >= len(group_ids):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Not enough groups to keep every split non-empty")

    return (
        set(group_ids[test_count + validation_count :]),
        set(group_ids[test_count : test_count + validation_count]),
        set(group_ids[:test_count]),
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    records = load_jsonl(DATASET_PATH)
    train, validation, test = split_by_group(
        records,
        validation_ratio=VALIDATION_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
        stratify_field=STRATIFY_FIELD,
    )

    write_jsonl(OUTPUT_DIR / "train.jsonl", train)
    write_jsonl(OUTPUT_DIR / "validation.jsonl", validation)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test)

    print(f"train={len(train)}, validation={len(validation)}, test={len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
