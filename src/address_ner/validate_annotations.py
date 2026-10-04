"""Validate character-span address annotations before BIO conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .labels import ENTITY_TYPES


# Configuration: edit this path before running this module.
DATASET_PATH = Path("data/synthetic/annotations-v2.jsonl")


ALLOWED_STATUSES = {"needs_review", "approved", "rejected"}
ALLOWED_SOURCES = {"manual", "synthetic", "real"}
REQUIRED_FIELDS = {
    "id": str,
    "text": str,
    "label": list,
    "annotation_status": str,
    "source": str,
}


@dataclass(frozen=True)
class AnnotationIssue:
    line_number: int
    record_id: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line_number} [{self.record_id}]: {self.message}"


def _record_id(record: Any) -> str:
    if isinstance(record, dict) and isinstance(record.get("id"), str):
        return record["id"]
    return "unknown-id"


def validate_annotation_record(
    record: Any, line_number: int = 1
) -> list[AnnotationIssue]:
    issues: list[AnnotationIssue] = []
    record_id = _record_id(record)
    if not isinstance(record, dict):
        return [AnnotationIssue(line_number, record_id, "record must be a JSON object")]

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            issues.append(AnnotationIssue(line_number, record_id, f"missing field '{field}'"))
        elif not isinstance(record[field], expected_type):
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    f"field '{field}' must be {expected_type.__name__}",
                )
            )
    if issues:
        return issues

    text = record["text"]
    if not record["id"].strip():
        issues.append(AnnotationIssue(line_number, record_id, "id must not be blank"))
    if not text.strip():
        issues.append(AnnotationIssue(line_number, record_id, "text must not be blank"))
    if record["annotation_status"] not in ALLOWED_STATUSES:
        issues.append(
            AnnotationIssue(
                line_number,
                record_id,
                f"annotation_status must be one of {sorted(ALLOWED_STATUSES)}",
            )
        )
    if record["source"] not in ALLOWED_SOURCES:
        issues.append(
            AnnotationIssue(
                line_number,
                record_id,
                f"source must be one of {sorted(ALLOWED_SOURCES)}",
            )
        )

    valid_spans: list[tuple[int, int, str, int]] = []
    for span_index, span in enumerate(record["label"]):
        if not isinstance(span, list) or len(span) != 3:
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    f"label {span_index} must be [start, end, entity_type]",
                )
            )
            continue

        start, end, entity_type = span
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    f"label {span_index} start/end must be integers",
                )
            )
            continue
        if not isinstance(entity_type, str) or entity_type not in ENTITY_TYPES:
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    f"label {span_index} has unknown entity type {entity_type!r}",
                )
            )
            continue
        if start < 0 or end > len(text) or start >= end:
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    f"label {span_index} has invalid range [{start}, {end}) for text length {len(text)}",
                )
            )
            continue
        if text[start:end] != text[start:end].strip():
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    f"label {span_index} includes leading or trailing whitespace",
                )
            )
        valid_spans.append((start, end, entity_type, span_index))

    sorted_spans = sorted(valid_spans)
    for previous, current in zip(sorted_spans, sorted_spans[1:]):
        if current[0] < previous[1]:
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    f"labels {previous[3]} and {current[3]} overlap",
                )
            )

    if valid_spans != sorted_spans:
        issues.append(
            AnnotationIssue(line_number, record_id, "labels must be sorted by start offset")
        )
    return issues


def load_annotation_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            if raw_line.strip():
                records.append(json.loads(raw_line))
    return records


def validate_annotation_jsonl(
    path: Path,
) -> tuple[int, list[AnnotationIssue]]:
    issues: list[AnnotationIssue] = []
    seen_ids: dict[str, int] = {}
    record_count = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            record_count += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                issues.append(
                    AnnotationIssue(
                        line_number, "unknown-id", f"invalid JSON: {error.msg}"
                    )
                )
                continue
            issues.extend(validate_annotation_record(record, line_number))
            record_id = _record_id(record)
            if record_id in seen_ids:
                issues.append(
                    AnnotationIssue(
                        line_number,
                        record_id,
                        f"duplicate id; first seen on line {seen_ids[record_id]}",
                    )
                )
            else:
                seen_ids[record_id] = line_number
    return record_count, issues


def main() -> int:
    record_count, issues = validate_annotation_jsonl(DATASET_PATH)
    if issues:
        print(f"FAILED: {len(issues)} issue(s) in {record_count} record(s)")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"OK: {record_count} valid annotation record(s) in {DATASET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
