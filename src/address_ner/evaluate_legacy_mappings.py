"""Evaluate legacy mapping decisions against manually reviewed cases."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .build_legacy_mappings import LegacyMapping, load_legacy_mappings
from .legacy_resolver import classify_legacy_mapping_candidates


# Configuration: edit these paths before evaluating legacy mappings.
DATASET_PATH = Path("data/golden/legacy_mapping_cases.seed.jsonl")
MAPPINGS_PATH = Path("data/catalog/legacy_mappings_2025.csv")
OUTPUT_PATH = Path("reports/legacy_mappings.seed.json")


def load_mapping_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    valid_statuses = {"WARNING", "AMBIGUOUS", "REJECTED"}
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            case = json.loads(raw_line)
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"line {line_number}: id must be non-empty")
            if case_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id {case_id!r}")
            legacy_code = case.get("legacy_ward_code")
            if not isinstance(legacy_code, str) or not legacy_code:
                raise ValueError(
                    f"line {line_number}: legacy_ward_code must be non-empty"
                )
            expected = case.get("expected")
            if not isinstance(expected, dict) or expected.get("status") not in valid_statuses:
                raise ValueError(
                    f"line {line_number}: expected.status must be one of "
                    f"{sorted(valid_statuses)}"
                )
            candidates = expected.get("candidate_ward_codes", [])
            if not isinstance(candidates, list) or not all(
                isinstance(code, str) for code in candidates
            ):
                raise ValueError(
                    f"line {line_number}: candidate_ward_codes must be strings"
                )
            categories = case.get("categories", [])
            if not isinstance(categories, list) or not all(
                isinstance(category, str) for category in categories
            ):
                raise ValueError(
                    f"line {line_number}: categories must be strings"
                )
            seen_ids.add(case_id)
            cases.append(case)
    return cases


def _score_items(items: list[dict[str, Any]]) -> dict[str, int | float]:
    num_cases = len(items)
    num_expected_accept = sum(item["expected_accept"] for item in items)
    num_accepted = sum(item["accepted"] for item in items)
    num_correct_accepts = sum(item["accepted_correct"] for item in items)
    num_false_accepts = sum(item["false_accept"] for item in items)
    num_correct = sum(item["correct"] for item in items)
    return {
        "num_cases": num_cases,
        "num_expected_accept": num_expected_accept,
        "num_accepted": num_accepted,
        "num_correct_accepts": num_correct_accepts,
        "num_false_accepts": num_false_accepts,
        "accept_precision": (
            num_correct_accepts / num_accepted if num_accepted else 0.0
        ),
        "coverage": (
            num_correct_accepts / num_expected_accept
            if num_expected_accept
            else 0.0
        ),
        "decision_accuracy": num_correct / num_cases if num_cases else 0.0,
    }


def evaluate_mapping_cases(
    mappings: list[LegacyMapping],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_legacy_code: dict[str, list[LegacyMapping]] = defaultdict(list)
    for mapping in mappings:
        by_legacy_code[mapping.legacy_ward_code].append(mapping)

    evaluated: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        candidates = by_legacy_code.get(case["legacy_ward_code"], [])
        status, selected = classify_legacy_mapping_candidates(candidates)
        current_ward_code = (
            selected.current_ward_code if selected is not None else None
        )
        candidate_codes = sorted(
            {mapping.current_ward_code for mapping in candidates}
        )
        expected = case["expected"]
        expected_accept = expected["status"] == "WARNING"
        accepted = status == "WARNING"
        status_match = status == expected["status"]
        current_match = (
            expected.get("current_ward_code") == current_ward_code
            if "current_ward_code" in expected
            else True
        )
        candidates_match = (
            sorted(expected["candidate_ward_codes"]) == candidate_codes
            if "candidate_ward_codes" in expected
            else True
        )
        accepted_correct = (
            accepted and expected_accept and current_match and candidates_match
        )
        item = {
            "id": case["id"],
            "expected_accept": expected_accept,
            "accepted": accepted,
            "accepted_correct": accepted_correct,
            "false_accept": accepted and not accepted_correct,
            "correct": status_match and current_match and candidates_match,
            "expected": expected,
            "actual": {
                "status": status,
                "current_ward_code": current_ward_code,
                "candidate_ward_codes": candidate_codes,
            },
        }
        evaluated.append(item)
        for category in case.get("categories", []):
            by_category[category].append(item)

    return {
        **_score_items(evaluated),
        "per_category": {
            category: _score_items(items)
            for category, items in sorted(by_category.items())
        },
        "failed_cases": [item for item in evaluated if not item["correct"]],
    }


def main() -> int:
    report = {
        "dataset": str(DATASET_PATH.resolve()),
        "mappings": str(MAPPINGS_PATH.resolve()),
        "evaluation": evaluate_mapping_cases(
            load_legacy_mappings(MAPPINGS_PATH.resolve()),
            load_mapping_cases(DATASET_PATH.resolve()),
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output = OUTPUT_PATH.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"wrote legacy mapping evaluation report to {output}")
    print(json.dumps(report["evaluation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
