"""Prepare and finalize a blinded legacy-mapping holdout review set."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .build_legacy_mappings import (
    LegacyMapping,
    ProvinceSuccessor,
    build_province_successors,
    load_legacy_mappings,
    load_province_comparisons,
)
from .evaluate_legacy_mappings import load_mapping_cases
from .legacy_resolver import (
    LegacyComparison,
    classify_legacy_mapping_candidates,
    load_legacy_comparisons,
)
from .prepare_annotation import load_catalog
from .resolver import fold_accents


# Configuration: set MODE to "prepare" or "finalize", then edit paths if needed.
MODE = "prepare"
WARD_COMPARISON_PATH = Path("data/catalog/ward_comparison_2025.csv")
PROVINCE_COMPARISON_PATH = Path("data/catalog/province_comparison_2025.csv")
CURRENT_CATALOG_PATH = Path("data/catalog/admin_units_2025.csv")
MAPPINGS_PATH = Path("data/catalog/legacy_mappings_2025.csv")
EXCLUDED_CASES_PATH = Path("data/golden/legacy_mapping_cases.seed.jsonl")
PER_STRATUM = 10
RANDOM_SEED = 20250920
REVIEW_PATH = Path("data/review/legacy_mapping_holdout.blind.jsonl")
MANIFEST_PATH = Path("data/review/legacy_mapping_holdout.manifest.json")
FINAL_OUTPUT_PATH = Path("data/golden/legacy_mapping_cases.holdout.jsonl")


STRATA = ("direct", "note_full", "split", "source_hint", "unresolved")
ADMIN_PREFIX = re.compile(
    r"^(phường|xã|thị trấn|đặc khu)\s+", flags=re.IGNORECASE
)


def mapping_stratum(mappings: list[LegacyMapping]) -> str | None:
    if not mappings:
        return "unresolved"
    methods = {mapping.mapping_method for mapping in mappings}
    status, _ = classify_legacy_mapping_candidates(mappings)
    if "NOTE_PARTIAL" in methods:
        return "split"
    if methods == {"DIRECT_CODE"}:
        return "direct"
    if status == "WARNING" and "NOTE_FULL" in methods:
        return "note_full"
    if methods == {"SOURCE_NOTE_HINT"}:
        return "source_hint"
    return None


def select_holdout_records(
    comparisons: list[LegacyComparison],
    mappings: list[LegacyMapping],
    excluded_codes: set[str],
    per_stratum: int,
    random_seed: int,
) -> list[tuple[str, LegacyComparison]]:
    mappings_by_code: dict[str, list[LegacyMapping]] = defaultdict(list)
    for mapping in mappings:
        mappings_by_code[mapping.legacy_ward_code].append(mapping)

    unique_records = {
        record.legacy_ward_code: record
        for record in comparisons
        if record.legacy_ward_code
    }
    by_stratum: dict[str, list[LegacyComparison]] = defaultdict(list)
    for code, record in unique_records.items():
        if code in excluded_codes:
            continue
        stratum = mapping_stratum(mappings_by_code.get(code, []))
        if stratum is not None:
            by_stratum[stratum].append(record)

    rng = random.Random(random_seed)
    selected: list[tuple[str, LegacyComparison]] = []
    for stratum in STRATA:
        candidates = sorted(
            by_stratum[stratum], key=lambda record: record.legacy_ward_code
        )
        if len(candidates) < per_stratum:
            raise ValueError(
                f"stratum {stratum!r} has only {len(candidates)} candidates; "
                f"requested {per_stratum}"
            )
        selected.extend(
            (stratum, record)
            for record in rng.sample(candidates, per_stratum)
        )
    rng.shuffle(selected)
    return selected


def _related_current_records(
    source: LegacyComparison,
    comparisons: list[LegacyComparison],
    mappings: list[LegacyMapping],
    province_successors: dict[str, ProvinceSuccessor],
) -> tuple[list[dict[str, str]], bool]:
    successor = province_successors.get(source.legacy_province_code)
    target_province_code = (
        successor.current_province_code if successor is not None else ""
    )
    mapped_codes = {
        mapping.current_ward_code
        for mapping in mappings
        if mapping.legacy_ward_code == source.legacy_ward_code
    }
    if source.current_ward_code:
        mapped_codes.add(source.current_ward_code)

    full_name = fold_accents(source.legacy_ward_name)
    short_name = fold_accents(ADMIN_PREFIX.sub("", source.legacy_ward_name))
    evidence_by_code: dict[str, dict[str, str]] = {}
    for target in comparisons:
        if not target.current_ward_code:
            continue
        folded_note = fold_accents(target.note)
        name_mentioned = bool(
            folded_note
            and (
                full_name in folded_note
                or (
                    len(short_name) >= 4
                    and re.search(
                        rf"(?<!\w){re.escape(short_name)}(?!\w)",
                        folded_note,
                    )
                )
            )
        )
        selected_by_code = target.current_ward_code in mapped_codes
        same_current_province = (
            not target_province_code
            or target.current_province_code == target_province_code
        )
        if not same_current_province or not (name_mentioned or selected_by_code):
            continue
        evidence_by_code[target.current_ward_code] = {
            "current_ward_code": target.current_ward_code,
            "current_ward_name": target.current_ward_name,
            "current_province_code": target.current_province_code,
            "current_province_name": target.current_province_name,
            "note": target.note,
            "source_url": target.source_url,
        }

    evidence = sorted(
        evidence_by_code.values(),
        key=lambda item: item["current_ward_code"],
    )
    limit = 20
    return evidence[:limit], len(evidence) > limit


def build_blind_cases(
    selected: list[tuple[str, LegacyComparison]],
    comparisons: list[LegacyComparison],
    mappings: list[LegacyMapping],
    province_successors: dict[str, ProvinceSuccessor],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    cases: list[dict[str, Any]] = []
    manifest_items: list[dict[str, str]] = []
    for index, (stratum, source) in enumerate(selected, start=1):
        case_id = f"legacy_holdout_{index:03d}"
        evidence, truncated = _related_current_records(
            source,
            comparisons,
            mappings,
            province_successors,
        )
        cases.append(
            {
                "id": case_id,
                "legacy": {
                    "province_code": source.legacy_province_code,
                    "province_name": source.legacy_province_name,
                    "district_code": source.legacy_district_code,
                    "district_name": source.legacy_district_name,
                    "ward_code": source.legacy_ward_code,
                    "ward_name": source.legacy_ward_name,
                },
                "source_note": source.note,
                "related_current_evidence": evidence,
                "evidence_truncated": truncated,
                "source_url": source.source_url,
                "review": {
                    "status": None,
                    "current_ward_code": None,
                    "candidate_ward_codes": [],
                    "reviewer_note": "",
                },
            }
        )
        manifest_items.append(
            {
                "id": case_id,
                "legacy_ward_code": source.legacy_ward_code,
                "stratum": stratum,
            }
        )
    return cases, manifest_items


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def finalize_review_cases(
    cases: list[dict[str, Any]],
    manifest_items: list[dict[str, str]],
) -> list[dict[str, Any]]:
    manifest_by_id = {item["id"]: item for item in manifest_items}
    finalized: list[dict[str, Any]] = []
    valid_statuses = {"WARNING", "AMBIGUOUS", "REJECTED"}
    for case in cases:
        case_id = case["id"]
        if case_id not in manifest_by_id:
            raise ValueError(f"case {case_id!r} is missing from manifest")
        review = case.get("review", {})
        status = review.get("status")
        current_code = review.get("current_ward_code")
        candidate_codes = sorted(set(review.get("candidate_ward_codes", [])))
        if status not in valid_statuses:
            raise ValueError(f"case {case_id!r} has not been reviewed")
        if status == "WARNING":
            if not isinstance(current_code, str) or not current_code:
                raise ValueError(
                    f"case {case_id!r}: WARNING requires current_ward_code"
                )
            if current_code not in candidate_codes:
                raise ValueError(
                    f"case {case_id!r}: selected code must be a candidate"
                )
        elif current_code is not None:
            raise ValueError(
                f"case {case_id!r}: only WARNING may select current_ward_code"
            )
        if status == "AMBIGUOUS" and not candidate_codes:
            raise ValueError(
                f"case {case_id!r}: AMBIGUOUS requires candidates"
            )
        if status == "REJECTED" and candidate_codes:
            raise ValueError(
                f"case {case_id!r}: REJECTED cannot contain candidates"
            )

        manifest = manifest_by_id[case_id]
        finalized.append(
            {
                "id": case_id,
                "legacy_ward_code": manifest["legacy_ward_code"],
                "categories": ["holdout", manifest["stratum"]],
                "expected": {
                    "status": status,
                    "current_ward_code": current_code,
                    "candidate_ward_codes": candidate_codes,
                },
                "review_note": review.get("reviewer_note", ""),
            }
        )
    return finalized


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def prepare() -> int:
    comparisons = load_legacy_comparisons(WARD_COMPARISON_PATH.resolve())
    mappings = load_legacy_mappings(MAPPINGS_PATH.resolve())
    excluded_codes = {
        case["legacy_ward_code"]
        for case in load_mapping_cases(EXCLUDED_CASES_PATH.resolve())
    }
    selected = select_holdout_records(
        comparisons,
        mappings,
        excluded_codes,
        PER_STRATUM,
        RANDOM_SEED,
    )
    province_successors = build_province_successors(
        load_province_comparisons(PROVINCE_COMPARISON_PATH.resolve()),
        load_catalog(CURRENT_CATALOG_PATH.resolve()),
    )
    cases, manifest_items = build_blind_cases(
        selected,
        comparisons,
        mappings,
        province_successors,
    )
    write_jsonl(REVIEW_PATH.resolve(), cases)
    manifest = {
        "random_seed": RANDOM_SEED,
        "per_stratum": PER_STRATUM,
        "num_cases": len(cases),
        "excluded_dataset": str(EXCLUDED_CASES_PATH.resolve()),
        "items": manifest_items,
    }
    manifest_path = MANIFEST_PATH.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} blinded cases to {REVIEW_PATH.resolve()}")
    print(f"wrote holdout manifest to {manifest_path}")
    return 0


def finalize() -> int:
    cases = _read_jsonl(REVIEW_PATH.resolve())
    manifest = json.loads(MANIFEST_PATH.resolve().read_text(encoding="utf-8"))
    finalized = finalize_review_cases(cases, manifest["items"])
    write_jsonl(FINAL_OUTPUT_PATH.resolve(), finalized)
    print(
        f"wrote {len(finalized)} reviewed holdout cases "
        f"to {FINAL_OUTPUT_PATH.resolve()}"
    )
    return 0


def main() -> int:
    if MODE == "prepare":
        return prepare()
    if MODE == "finalize":
        return finalize()
    raise ValueError("MODE must be either 'prepare' or 'finalize'")


if __name__ == "__main__":
    raise SystemExit(main())
