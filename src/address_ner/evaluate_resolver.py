"""Evaluate and calibrate the administrative-unit resolver on fixed cases."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable

from .prepare_annotation import AdminUnit, load_catalog
from .resolver import AdminResolver


# Configuration: edit these values before evaluating the resolver.
DATASET_PATH = Path("data/golden/resolver_cases.seed.jsonl")
CATALOG_PATH = Path("data/catalog/admin_units_2025.csv")
WARD_THRESHOLD = 0.88
PROVINCE_THRESHOLD = 0.86
FUZZY_MARGIN = 0.03
CALIBRATE = False
WARD_THRESHOLD_GRID = [0.80, 0.82, 0.84, 0.86, 0.88]
PROVINCE_THRESHOLD_GRID = [0.78, 0.82, 0.86]
MARGIN_GRID = [0.03, 0.06, 0.09]
OUTPUT_PATH: Path | None = Path("reports/resolver.seed.json")


ACCEPTED_STATUSES = {"PASS", "WARNING"}


def load_resolution_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            case = json.loads(raw_line)
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"line {line_number}: id must be a non-empty string")
            if case_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id {case_id!r}")
            if not isinstance(case.get("entities"), list):
                raise ValueError(f"line {line_number}: entities must be a list")
            expected = case.get("expected")
            if not isinstance(expected, dict) or not isinstance(
                expected.get("accept"), bool
            ):
                raise ValueError(
                    f"line {line_number}: expected.accept must be a boolean"
                )
            categories = case.get("categories", [])
            if not isinstance(categories, list) or not all(
                isinstance(category, str) for category in categories
            ):
                raise ValueError(f"line {line_number}: categories must be strings")
            seen_ids.add(case_id)
            cases.append(case)
    return cases


def _resolved_code(result: dict[str, Any], component: str) -> str | None:
    resolved = result["current"].get(component)
    return str(resolved["code"]) if resolved else None


def _score_items(items: list[dict[str, Any]]) -> dict[str, int | float]:
    num_cases = len(items)
    num_expected_accept = sum(item["expected_accept"] for item in items)
    num_accepted = sum(item["accepted"] for item in items)
    num_correct_accepts = sum(item["accepted_correct"] for item in items)
    num_false_accepts = sum(item["false_accept"] for item in items)
    num_correct_decisions = sum(item["correct"] for item in items)
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
        "decision_accuracy": (
            num_correct_decisions / num_cases if num_cases else 0.0
        ),
    }


def evaluate_resolution_cases(
    resolver: AdminResolver, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in cases:
        result = resolver.resolve(case["entities"])
        expected = case["expected"]
        accepted = result["status"] in ACCEPTED_STATUSES
        expected_accept = expected["accept"]
        ward_code = _resolved_code(result, "ward")
        province_code = _resolved_code(result, "province")
        codes_match = all(
            expected.get(field) is None or expected.get(field) == actual
            for field, actual in (
                ("ward_code", ward_code),
                ("province_code", province_code),
            )
        )
        status_match = (
            expected.get("status") is None
            or expected["status"] == result["status"]
        )
        accepted_correct = accepted and expected_accept and codes_match
        false_accept = accepted and not accepted_correct
        correct = (
            accepted_correct
            if expected_accept
            else (not accepted and status_match)
        )
        item = {
            "id": case["id"],
            "expected_accept": expected_accept,
            "accepted": accepted,
            "accepted_correct": accepted_correct,
            "false_accept": false_accept,
            "correct": correct,
            "expected": expected,
            "actual": {
                "status": result["status"],
                "ward_code": ward_code,
                "province_code": province_code,
                "resolver_confidence": result["resolver_confidence"],
                "candidate_wards": result["candidate_wards"],
                "issues": result["issues"],
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


def calibrate_thresholds(
    units: list[AdminUnit],
    cases: list[dict[str, Any]],
    ward_thresholds: Iterable[float],
    province_thresholds: Iterable[float],
    margins: Iterable[float],
) -> list[dict[str, Any]]:
    resolver = AdminResolver(units)
    candidates: list[dict[str, Any]] = []
    for ward_threshold, province_threshold, margin in product(
        ward_thresholds, province_thresholds, margins
    ):
        resolver.ward_fuzzy_threshold = ward_threshold
        resolver.province_fuzzy_threshold = province_threshold
        resolver.fuzzy_margin = margin
        report = evaluate_resolution_cases(resolver, cases)
        candidates.append(
            {
                "ward_fuzzy_threshold": ward_threshold,
                "province_fuzzy_threshold": province_threshold,
                "fuzzy_margin": margin,
                **{
                    key: report[key]
                    for key in (
                        "num_false_accepts",
                        "num_correct_accepts",
                        "accept_precision",
                        "coverage",
                        "decision_accuracy",
                    )
                },
            }
        )

    return sorted(
        candidates,
        key=lambda item: (
            item["num_false_accepts"],
            -item["num_correct_accepts"],
            -item["decision_accuracy"],
            -item["accept_precision"],
            -item["ward_fuzzy_threshold"],
            -item["province_fuzzy_threshold"],
            -item["fuzzy_margin"],
        ),
    )


def _float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    units = load_catalog(CATALOG_PATH.resolve())
    cases = load_resolution_cases(DATASET_PATH.resolve())
    resolver = AdminResolver(
        units,
        ward_fuzzy_threshold=WARD_THRESHOLD,
        province_fuzzy_threshold=PROVINCE_THRESHOLD,
        fuzzy_margin=FUZZY_MARGIN,
    )
    report: dict[str, Any] = {
        "dataset": str(DATASET_PATH.resolve()),
        "configuration": {
            "ward_fuzzy_threshold": WARD_THRESHOLD,
            "province_fuzzy_threshold": PROVINCE_THRESHOLD,
            "fuzzy_margin": FUZZY_MARGIN,
        },
        "evaluation": evaluate_resolution_cases(resolver, cases),
    }
    if CALIBRATE:
        ranking = calibrate_thresholds(
            units,
            cases,
            WARD_THRESHOLD_GRID,
            PROVINCE_THRESHOLD_GRID,
            MARGIN_GRID,
        )
        report["calibration"] = {
            "num_configurations": len(ranking),
            "recommended": ranking[0] if ranking else None,
            "top_configurations": ranking[:10],
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if OUTPUT_PATH:
        output_path = OUTPUT_PATH.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote resolver evaluation report to {output_path}")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
