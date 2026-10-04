"""Validate word-level JSONL records before PhoBERT tokenization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .labels import ENTITY_TYPES, LABEL2ID


# Configuration: edit this path before running this module.
DATASET_PATH = Path("data/sample.jsonl")


REQUIRED_FIELDS = {
    "id": str,
    "group_id": str,
    "raw_text": str,
    "tokens": list,
    "ner_tags": list,
    "source": str,
}

ALLOWED_SOURCES = {"manual", "synthetic", "real"}


@dataclass(frozen=True)
class ValidationIssue:
    line_number: int
    record_id: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line_number} [{self.record_id}]: {self.message}"


def _record_id(record: Any) -> str:
    if isinstance(record, dict) and isinstance(record.get("id"), str):
        return record["id"]
    return "unknown-id"


def validate_record(record: Any, line_number: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    record_id = _record_id(record)

    if not isinstance(record, dict):
        return [ValidationIssue(line_number, record_id, "record must be a JSON object")]

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            issues.append(ValidationIssue(line_number, record_id, f"missing field '{field}'"))
        elif not isinstance(record[field], expected_type):
            issues.append(
                ValidationIssue(
                    line_number,
                    record_id,
                    f"field '{field}' must be {expected_type.__name__}",
                )
            )

    if issues:
        return issues

    if not record["id"].strip():
        issues.append(ValidationIssue(line_number, record_id, "id must not be blank"))
    if not record["group_id"].strip():
        issues.append(ValidationIssue(line_number, record_id, "group_id must not be blank"))
    if not record["raw_text"].strip():
        issues.append(ValidationIssue(line_number, record_id, "raw_text must not be blank"))
    if record["source"] not in ALLOWED_SOURCES:
        issues.append(
            ValidationIssue(
                line_number,
                record_id,
                f"source must be one of {sorted(ALLOWED_SOURCES)}",
            )
        )

    tokens = record["tokens"]
    tags = record["ner_tags"]

    if not tokens:
        issues.append(ValidationIssue(line_number, record_id, "tokens must not be empty"))
    if len(tokens) != len(tags):
        issues.append(
            ValidationIssue(
                line_number,
                record_id,
                f"token count {len(tokens)} does not match tag count {len(tags)}",
            )
        )
        return issues

    for index, token in enumerate(tokens):
        if not isinstance(token, str) or not token.strip():
            issues.append(
                ValidationIssue(line_number, record_id, f"token {index} must be a non-blank string")
            )

    previous_prefix = "O"
    previous_entity: str | None = None

    for index, tag in enumerate(tags):
        if not isinstance(tag, str):
            issues.append(ValidationIssue(line_number, record_id, f"tag {index} must be a string"))
            previous_prefix, previous_entity = "O", None
            continue

        if tag not in LABEL2ID:
            issues.append(ValidationIssue(line_number, record_id, f"unknown tag '{tag}' at index {index}"))
            previous_prefix, previous_entity = "O", None
            continue

        if tag == "O":
            previous_prefix, previous_entity = "O", None
            continue

        prefix, entity = tag.split("-", maxsplit=1)
        if entity not in ENTITY_TYPES:
            issues.append(
                ValidationIssue(line_number, record_id, f"unknown entity type '{entity}' at index {index}")
            )

        if prefix == "I" and not (
            previous_prefix in {"B", "I"} and previous_entity == entity
        ):
            issues.append(
                ValidationIssue(
                    line_number,
                    record_id,
                    f"'{tag}' at index {index} does not continue a {entity} entity",
                )
            )

        previous_prefix, previous_entity = prefix, entity

    return issues


def validate_jsonl(path: Path) -> tuple[int, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    record_count = 0
    seen_ids: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue

            record_count += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                issues.append(
                    ValidationIssue(line_number, "unknown-id", f"invalid JSON: {error.msg}")
                )
                continue

            issues.extend(validate_record(record, line_number))

            record_id = _record_id(record)
            if record_id != "unknown-id":
                if record_id in seen_ids:
                    issues.append(
                        ValidationIssue(
                            line_number,
                            record_id,
                            f"duplicate id; first seen on line {seen_ids[record_id]}",
                        )
                    )
                else:
                    seen_ids[record_id] = line_number

    return record_count, issues


def main() -> int:
    record_count, issues = validate_jsonl(DATASET_PATH)
    if issues:
        print(f"FAILED: {len(issues)} issue(s) in {record_count} record(s)")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"OK: {record_count} valid record(s) in {DATASET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
