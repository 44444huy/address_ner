"""Build conservative old-to-current ward mappings from official NSO notes."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .legacy_resolver import LegacyComparison, load_legacy_comparisons
from .prepare_annotation import AdminUnit, load_catalog
from .resolver import fold_accents


# Configuration: edit these paths before rebuilding legacy mappings.
WARD_COMPARISON_PATH = Path("data/catalog/ward_comparison_2025.csv")
PROVINCE_COMPARISON_PATH = Path("data/catalog/province_comparison_2025.csv")
CURRENT_CATALOG_PATH = Path("data/catalog/admin_units_2025.csv")
OUTPUT_PATH = Path("data/catalog/legacy_mappings_2025.csv")
REPORT_PATH = Path("data/catalog/legacy_mappings.report.json")


@dataclass(frozen=True)
class ProvinceComparison:
    legacy_province_code: str
    legacy_province_name: str
    legacy_effective_from: str
    current_province_code: str
    current_province_name: str
    current_effective_from: str
    note: str
    source_url: str


@dataclass(frozen=True)
class ProvinceSuccessor:
    current_province_code: str
    current_province_name: str
    mapping_method: str


@dataclass(frozen=True)
class LegacyMapping:
    legacy_province_code: str
    legacy_district_code: str
    legacy_ward_code: str
    legacy_ward_name: str
    current_province_code: str
    current_province_name: str
    current_ward_code: str
    current_ward_name: str
    mapping_method: str
    evidence: str
    source_url: str


def _read_dataclasses(path: Path, record_type: type) -> list:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            record_type(
                **{
                    field: row[field]
                    for field in record_type.__dataclass_fields__
                }
            )
            for row in csv.DictReader(file)
        ]


def load_province_comparisons(path: Path) -> list[ProvinceComparison]:
    return _read_dataclasses(path, ProvinceComparison)


def load_legacy_mappings(path: Path) -> list[LegacyMapping]:
    return _read_dataclasses(path, LegacyMapping)


def build_province_successors(
    comparisons: Iterable[ProvinceComparison],
    current_units: Iterable[AdminUnit],
) -> dict[str, ProvinceSuccessor]:
    current_by_code: dict[str, AdminUnit] = {}
    current_by_name: dict[str, AdminUnit] = {}
    for unit in current_units:
        current_by_code[unit.province_code] = unit
        current_by_name[fold_accents(unit.province_name)] = unit

    successors: dict[str, ProvinceSuccessor] = {}
    for comparison in comparisons:
        if comparison.current_province_code:
            successors[comparison.legacy_province_code] = ProvinceSuccessor(
                comparison.current_province_code,
                comparison.current_province_name,
                "DIRECT_PROVINCE_ROW",
            )
            continue

        folded_note = fold_accents(comparison.note)
        destination = folded_note.rsplit("ten goi la", maxsplit=1)[-1]
        note_matches = [
            unit
            for normalized_name, unit in current_by_name.items()
            if normalized_name and normalized_name in destination
        ]
        unique_note_matches = {
            unit.province_code: unit for unit in note_matches
        }
        if len(unique_note_matches) == 1:
            unit = next(iter(unique_note_matches.values()))
            successors[comparison.legacy_province_code] = ProvinceSuccessor(
                unit.province_code,
                unit.province_name,
                "NOTE_PROVINCE_NAME",
            )
            continue

        same_code = current_by_code.get(comparison.legacy_province_code)
        if same_code is not None:
            successors[comparison.legacy_province_code] = ProvinceSuccessor(
                same_code.province_code,
                same_code.province_name,
                "SAME_PROVINCE_CODE",
            )

    return successors


_ACTION_PATTERN = re.compile(
    r"\b(sap nhap|hop nhat|nhap|doi ten tu|chuyen|dieu chinh)\b"
)
_FULL_PATTERN = re.compile(
    r"\b(sap nhap|hop nhat|nhap|toan bo|doi ten tu|chuyen toan bo)\b"
)
_PARTIAL_PATTERN = re.compile(
    r"\b(1 phan|mot phan|phan con lai|phan dien tich)\b"
)
_DESTINATION_PATTERN = re.compile(
    r"\b(vao|thanh|ten goi la|thanh lap)\s*$"
)
_SOURCE_NOTE_DESTINATION = re.compile(
    r"đóng mã do sáp nhập vào\s+"
    r"((?:phường|xã|thị trấn|đặc khu)\s+[^.;]+)",
    flags=re.IGNORECASE,
)


def _sentence_start(note: str, position: int) -> int:
    return max(note.rfind(".", 0, position), note.rfind(";", 0, position)) + 1


def _mapping_method(note: str, mention_start: int) -> str | None:
    sentence = fold_accents(note[_sentence_start(note, mention_start):mention_start])
    if _DESTINATION_PATTERN.search(sentence[-80:]):
        return None
    if _ACTION_PATTERN.search(sentence) is None:
        return None

    partial_positions = [match.start() for match in _PARTIAL_PATTERN.finditer(sentence)]
    full_positions = [match.start() for match in _FULL_PATTERN.finditer(sentence)]
    last_partial = max(partial_positions, default=-1)
    last_full = max(full_positions, default=-1)
    if last_partial > last_full:
        return "NOTE_PARTIAL"
    if last_full >= 0:
        return "NOTE_FULL"
    return None


def mappings_from_note(
    target: LegacyComparison,
    legacy_candidates: Iterable[LegacyComparison],
) -> list[LegacyMapping]:
    if not target.current_ward_code or not target.note:
        return []

    note = target.note.casefold()
    mappings: list[LegacyMapping] = []
    seen: set[tuple[str, str, str]] = set()
    candidates_by_name: dict[str, list[LegacyComparison]] = defaultdict(list)
    for candidate in legacy_candidates:
        if candidate.legacy_ward_name:
            candidates_by_name[candidate.legacy_ward_name.casefold()].append(
                candidate
            )

    for ward_name, same_name_candidates in candidates_by_name.items():
        if not ward_name:
            continue
        pattern = re.compile(
            rf"(?<!\w){re.escape(ward_name)}(?!\w)"
        )
        for mention in pattern.finditer(note):
            method = _mapping_method(target.note, mention.start())
            if method is None:
                continue
            candidates = _disambiguate_same_name_candidates(
                target,
                same_name_candidates,
                mention.start(),
            )
            if len(candidates) != 1:
                continue
            candidate = candidates[0]
            key = (
                candidate.legacy_ward_code,
                target.current_ward_code,
                method,
            )
            if key in seen:
                continue
            seen.add(key)
            mappings.append(
                LegacyMapping(
                    legacy_province_code=candidate.legacy_province_code,
                    legacy_district_code=candidate.legacy_district_code,
                    legacy_ward_code=candidate.legacy_ward_code,
                    legacy_ward_name=candidate.legacy_ward_name,
                    current_province_code=target.current_province_code,
                    current_province_name=target.current_province_name,
                    current_ward_code=target.current_ward_code,
                    current_ward_name=target.current_ward_name,
                    mapping_method=method,
                    evidence=target.note,
                    source_url=target.source_url,
                )
            )
    return mappings


def _disambiguate_same_name_candidates(
    target: LegacyComparison,
    candidates: list[LegacyComparison],
    mention_start: int,
) -> list[LegacyComparison]:
    if len(candidates) <= 1:
        return candidates

    sentence_start = _sentence_start(target.note, mention_start)
    sentence_end = target.note.find(".", mention_start)
    if sentence_end < 0:
        sentence_end = len(target.note)
    context = fold_accents(
        target.note[sentence_start:min(sentence_end, mention_start + 240)]
    )

    province_matches = [
        candidate
        for candidate in candidates
        if fold_accents(candidate.legacy_province_name) in context
    ]
    if province_matches:
        candidates = province_matches

    district_matches = [
        candidate
        for candidate in candidates
        if fold_accents(candidate.legacy_district_name) in context
    ]
    if district_matches:
        candidates = district_matches

    if len(candidates) > 1 and target.legacy_district_code:
        same_district = [
            candidate
            for candidate in candidates
            if candidate.legacy_district_code == target.legacy_district_code
        ]
        if same_district:
            candidates = same_district

    return candidates


def mapping_from_source_note(
    source: LegacyComparison,
    province_successors: dict[str, ProvinceSuccessor],
    current_by_province_and_name: dict[
        tuple[str, str], dict[str, LegacyComparison]
    ],
) -> LegacyMapping | None:
    if not source.legacy_ward_code or not source.note:
        return None
    match = _SOURCE_NOTE_DESTINATION.search(source.note)
    if match is None:
        return None

    destination_name = re.sub(
        r"\s+mới\s*$", "", match.group(1).strip(), flags=re.IGNORECASE
    )
    successor = province_successors.get(source.legacy_province_code)
    if successor is None:
        return None
    targets = current_by_province_and_name.get(
        (
            successor.current_province_code,
            fold_accents(destination_name),
        ),
        {},
    )
    if len(targets) != 1:
        return None
    target = next(iter(targets.values()))
    return LegacyMapping(
        legacy_province_code=source.legacy_province_code,
        legacy_district_code=source.legacy_district_code,
        legacy_ward_code=source.legacy_ward_code,
        legacy_ward_name=source.legacy_ward_name,
        current_province_code=target.current_province_code,
        current_province_name=target.current_province_name,
        current_ward_code=target.current_ward_code,
        current_ward_name=target.current_ward_name,
        mapping_method="SOURCE_NOTE_HINT",
        evidence=source.note,
        source_url=source.source_url,
    )


def build_legacy_mappings(
    ward_comparisons: list[LegacyComparison],
    province_successors: dict[str, ProvinceSuccessor],
) -> list[LegacyMapping]:
    mappings: list[LegacyMapping] = []
    legacy_by_current_province: dict[str, dict[str, LegacyComparison]] = (
        defaultdict(dict)
    )
    current_by_province_and_name: dict[
        tuple[str, str], dict[str, LegacyComparison]
    ] = defaultdict(dict)

    for record in ward_comparisons:
        if record.current_ward_code:
            current_by_province_and_name[
                (
                    record.current_province_code,
                    fold_accents(record.current_ward_name),
                )
            ][record.current_ward_code] = record

    for record in ward_comparisons:
        if not record.legacy_ward_code:
            continue
        successor = province_successors.get(record.legacy_province_code)
        if successor is not None:
            legacy_by_current_province[successor.current_province_code][
                record.legacy_ward_code
            ] = record
        if record.current_ward_code:
            mappings.append(
                LegacyMapping(
                    legacy_province_code=record.legacy_province_code,
                    legacy_district_code=record.legacy_district_code,
                    legacy_ward_code=record.legacy_ward_code,
                    legacy_ward_name=record.legacy_ward_name,
                    current_province_code=record.current_province_code,
                    current_province_name=record.current_province_name,
                    current_ward_code=record.current_ward_code,
                    current_ward_name=record.current_ward_name,
                    mapping_method="DIRECT_CODE",
                    evidence="same NSO comparison row",
                    source_url=record.source_url,
                )
            )
        source_hint = mapping_from_source_note(
            record,
            province_successors,
            current_by_province_and_name,
        )
        if source_hint is not None:
            mappings.append(source_hint)

    for target in ward_comparisons:
        candidates = legacy_by_current_province.get(
            target.current_province_code, {}
        ).values()
        mappings.extend(mappings_from_note(target, candidates))

    unique = {
        (
            mapping.legacy_ward_code,
            mapping.current_ward_code,
            mapping.mapping_method,
        ): mapping
        for mapping in mappings
    }
    return sorted(
        unique.values(),
        key=lambda item: (
            item.legacy_ward_code,
            item.current_ward_code,
            item.mapping_method,
        ),
    )


def build_mapping_report(
    ward_comparisons: list[LegacyComparison],
    mappings: list[LegacyMapping],
    province_successors: dict[str, ProvinceSuccessor],
) -> dict[str, int]:
    legacy_codes = {
        row.legacy_ward_code for row in ward_comparisons if row.legacy_ward_code
    }
    by_legacy: dict[str, list[LegacyMapping]] = defaultdict(list)
    for mapping in mappings:
        by_legacy[mapping.legacy_ward_code].append(mapping)

    direct = {
        mapping.legacy_ward_code
        for mapping in mappings
        if mapping.mapping_method == "DIRECT_CODE"
    }
    note_full = {
        mapping.legacy_ward_code
        for mapping in mappings
        if mapping.mapping_method == "NOTE_FULL"
    }
    note_partial = {
        mapping.legacy_ward_code
        for mapping in mappings
        if mapping.mapping_method == "NOTE_PARTIAL"
    }
    source_hint = {
        mapping.legacy_ward_code
        for mapping in mappings
        if mapping.mapping_method == "SOURCE_NOTE_HINT"
    }
    ambiguous = 0
    for candidates in by_legacy.values():
        full_targets = {
            item.current_ward_code
            for item in candidates
            if item.mapping_method in {"DIRECT_CODE", "NOTE_FULL"}
        }
        uncertain_targets = {
            item.current_ward_code
            for item in candidates
            if item.mapping_method not in {"DIRECT_CODE", "NOTE_FULL"}
        }
        if len(full_targets) > 1 or (
            uncertain_targets and uncertain_targets - full_targets
        ):
            ambiguous += 1

    covered = set(by_legacy)
    return {
        "legacy_provinces_mapped": len(province_successors),
        "legacy_wards": len(legacy_codes),
        "mapping_rows": len(mappings),
        "direct_legacy_wards": len(direct),
        "note_full_legacy_wards": len(note_full),
        "note_partial_legacy_wards": len(note_partial),
        "source_hint_legacy_wards": len(source_hint),
        "partial_only_legacy_wards": len(note_partial - direct - note_full),
        "hint_only_legacy_wards": len(
            source_hint - direct - note_full - note_partial
        ),
        "ambiguous_legacy_wards": ambiguous,
        "unresolved_legacy_wards": len(legacy_codes - covered),
    }


def write_mappings(path: Path, mappings: list[LegacyMapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(LegacyMapping.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(mapping.__dict__ for mapping in mappings)


def main() -> int:
    ward_comparisons = load_legacy_comparisons(WARD_COMPARISON_PATH)
    province_successors = build_province_successors(
        load_province_comparisons(PROVINCE_COMPARISON_PATH),
        load_catalog(CURRENT_CATALOG_PATH),
    )
    mappings = build_legacy_mappings(ward_comparisons, province_successors)
    report = build_mapping_report(
        ward_comparisons, mappings, province_successors
    )
    write_mappings(OUTPUT_PATH, mappings)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
