"""Validate the fixed, never-train-on parser benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .validate_annotations import AnnotationIssue, validate_annotation_record


# Configuration: edit this path before validating a golden set.
DATASET_PATH = Path("data/golden/golden_cases.seed.jsonl")


GOLDEN_CATEGORIES = {
    "CURRENT_FULL",
    "LEGACY_FULL",
    "WITHOUT_ACCENTS",
    "ABBREVIATED",
    "MISSING_HOUSE_NUMBER",
    "MISSING_WARD",
    "MISSING_PROVINCE",
    "DUPLICATE_WARD_NAME",
    "MIXED_OLD_NEW",
    "PARTIAL",
    "NO_DELIMITERS",
    "INVALID",
}


def _record_id(record: Any) -> str:
    if isinstance(record, dict) and isinstance(record.get("id"), str):
        return record["id"]
    return "unknown-id"


def validate_golden_record(
    record: Any, line_number: int = 1
) -> list[AnnotationIssue]:
    """Validate one approved benchmark case and its evaluation categories."""
    issues = validate_annotation_record(record, line_number)
    record_id = _record_id(record)
    if not isinstance(record, dict):
        return issues

    categories = record.get("categories")
    if not isinstance(categories, list) or not categories:
        issues.append(
            AnnotationIssue(
                line_number, record_id, "categories must be a non-empty list"
            )
        )
        return issues

    seen_categories: set[str] = set()
    for category in categories:
        if not isinstance(category, str) or category not in GOLDEN_CATEGORIES:
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    f"unknown golden category {category!r}",
                )
            )
        elif category in seen_categories:
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    f"duplicate golden category {category!r}",
                )
            )
        else:
            seen_categories.add(category)

    if record.get("annotation_status") != "approved":
        issues.append(
            AnnotationIssue(
                line_number,
                record_id,
                "golden cases must have annotation_status='approved'",
            )
        )
    if record.get("source") == "synthetic":
        issues.append(
            AnnotationIssue(
                line_number,
                record_id,
                "synthetic examples must not be used as golden cases",
            )
        )

    labels = record.get("label")
    if isinstance(labels, list):
        is_invalid = "INVALID" in seen_categories
        if is_invalid and labels:
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    "INVALID cases must not contain expected entities",
                )
            )
        elif not is_invalid and not labels:
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    "non-INVALID cases must contain at least one expected entity",
                )
            )
        if is_invalid and len(seen_categories) > 1:
            issues.append(
                AnnotationIssue(
                    line_number,
                    record_id,
                    "INVALID must be the only category on a case",
                )
            )
    return issues


def validate_golden_jsonl(
    path: Path,
) -> tuple[int, Counter[str], list[AnnotationIssue]]:
    """Validate a JSONL benchmark and detect duplicate ids and texts."""
    issues: list[AnnotationIssue] = []
    category_counts: Counter[str] = Counter()
    seen_ids: dict[str, int] = {}
    seen_texts: dict[str, int] = {}
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

            issues.extend(validate_golden_record(record, line_number))
            record_id = _record_id(record)
            text = record.get("text") if isinstance(record, dict) else None
            categories = record.get("categories") if isinstance(record, dict) else None

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

            if isinstance(text, str):
                if text in seen_texts:
                    issues.append(
                        AnnotationIssue(
                            line_number,
                            record_id,
                            f"duplicate text; first seen on line {seen_texts[text]}",
                        )
                    )
                else:
                    seen_texts[text] = line_number
            if isinstance(categories, list):
                category_counts.update(
                    category
                    for category in categories
                    if isinstance(category, str) and category in GOLDEN_CATEGORIES
                )

    return record_count, category_counts, issues


def main() -> int:
    record_count, category_counts, issues = validate_golden_jsonl(DATASET_PATH)
    if issues:
        print(f"FAILED: {len(issues)} issue(s) in {record_count} golden case(s)")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"OK: {record_count} valid golden case(s) in {DATASET_PATH}")
    for category, count in sorted(category_counts.items()):
        print(f"- {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
