"""Resolve legacy province/district/ward entities through direct NSO mappings."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .resolver import fold_accents, normalize_admin_name


@dataclass(frozen=True)
class LegacyComparison:
    legacy_province_code: str
    legacy_province_name: str
    legacy_district_code: str
    legacy_district_name: str
    legacy_ward_code: str
    legacy_ward_name: str
    legacy_effective_from: str
    current_ward_code: str
    current_ward_name: str
    current_province_code: str
    current_province_name: str
    current_effective_from: str
    note: str
    source_url: str


def load_legacy_comparisons(path: Path) -> list[LegacyComparison]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            LegacyComparison(
                **{
                    field: row[field]
                    for field in LegacyComparison.__dataclass_fields__
                }
            )
            for row in csv.DictReader(file)
        ]


PREFIX_ALIASES = {
    "phường": ("P", "P."),
    "xã": ("X", "X."),
    "thị trấn": ("TT", "TT."),
    "quận": ("Q", "Q."),
    "huyện": ("H", "H."),
    "thị xã": ("TX", "TX."),
    "thành phố": ("TP", "TP."),
    "tỉnh": ("T", "T."),
}


def administrative_aliases(name: str) -> set[str]:
    aliases = {name}
    lowered = name.casefold()
    for prefix, abbreviations in PREFIX_ALIASES.items():
        marker = f"{prefix} "
        if not lowered.startswith(marker):
            continue
        short_name = name[len(marker) :]
        aliases.add(short_name)
        aliases.update(f"{abbreviation} {short_name}" for abbreviation in abbreviations)
        break
    return {normalize_admin_name(alias) for alias in aliases}


def _unit_type(name: str) -> str | None:
    match = re.match(
        r"^(Phường|Xã|Thị trấn|Đặc khu)\s+",
        name,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def _entity_confidence(entity: dict[str, Any] | None) -> float | None:
    if entity is None:
        return None
    value = entity.get("mean_confidence", entity.get("confidence"))
    return float(value) if value is not None else None


def _best_entity(
    entities: list[dict[str, Any]], entity_type: str
) -> dict[str, Any] | None:
    candidates = [entity for entity in entities if entity.get("type") == entity_type]
    return max(
        candidates,
        key=lambda entity: _entity_confidence(entity) or 0.0,
    ) if candidates else None


FULL_MAPPING_METHODS = {"DIRECT_CODE", "NOTE_FULL"}


def classify_legacy_mapping_candidates(
    mappings: list[Any],
) -> tuple[str, Any | None]:
    if not mappings:
        return "REJECTED", None

    full_mappings = [
        mapping
        for mapping in mappings
        if mapping.mapping_method in FULL_MAPPING_METHODS
    ]
    full_targets = {
        mapping.current_ward_code for mapping in full_mappings
    }
    uncertain_targets = {
        mapping.current_ward_code
        for mapping in mappings
        if mapping.mapping_method not in FULL_MAPPING_METHODS
    }
    if len(full_targets) == 1 and not (uncertain_targets - full_targets):
        chosen = min(
            full_mappings,
            key=lambda mapping: (
                mapping.mapping_method != "DIRECT_CODE",
                mapping.current_ward_code,
            ),
        )
        return "WARNING", chosen
    return "AMBIGUOUS", None


class LegacyAdminResolver:
    def __init__(
        self,
        records: list[LegacyComparison],
        mappings: list[Any] | None = None,
    ) -> None:
        self.has_mapping_catalog = mappings is not None
        self.mappings_by_legacy_code: dict[str, list[Any]] = defaultdict(list)
        for mapping in mappings or []:
            self.mappings_by_legacy_code[mapping.legacy_ward_code].append(mapping)
        self.ward_exact: dict[
            str, dict[str, LegacyComparison]
        ] = defaultdict(dict)
        self.ward_folded: dict[
            str, dict[str, LegacyComparison]
        ] = defaultdict(dict)
        self.district_exact: dict[
            str, dict[str, LegacyComparison]
        ] = defaultdict(dict)
        self.district_folded: dict[
            str, dict[str, LegacyComparison]
        ] = defaultdict(dict)
        self.province_exact: dict[
            str, dict[str, LegacyComparison]
        ] = defaultdict(dict)
        self.province_folded: dict[
            str, dict[str, LegacyComparison]
        ] = defaultdict(dict)

        for record in records:
            if not record.legacy_ward_code:
                continue
            self._register_aliases(
                self.ward_exact,
                self.ward_folded,
                administrative_aliases(record.legacy_ward_name),
                record.legacy_ward_code,
                record,
            )
            self._register_aliases(
                self.district_exact,
                self.district_folded,
                administrative_aliases(record.legacy_district_name),
                record.legacy_district_code,
                record,
            )
            self._register_aliases(
                self.province_exact,
                self.province_folded,
                administrative_aliases(record.legacy_province_name),
                record.legacy_province_code,
                record,
            )

    @staticmethod
    def _register_aliases(
        exact_index: dict[str, dict[str, LegacyComparison]],
        folded_index: dict[str, dict[str, LegacyComparison]],
        aliases: set[str],
        identity: str,
        record: LegacyComparison,
    ) -> None:
        if not identity:
            return
        for alias in aliases:
            exact_index[alias][identity] = record
            folded_index[fold_accents(alias)][identity] = record

    @staticmethod
    def _lookup(
        text: str,
        exact_index: dict[str, dict[str, LegacyComparison]],
        folded_index: dict[str, dict[str, LegacyComparison]],
    ) -> tuple[list[LegacyComparison], float | None]:
        exact = list(exact_index.get(normalize_admin_name(text), {}).values())
        if exact:
            return exact, 1.0
        folded = list(folded_index.get(fold_accents(text), {}).values())
        if folded:
            return folded, 0.9
        return [], None

    def resolve(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        ward_entity = _best_entity(entities, "WARD")
        district_entity = _best_entity(entities, "DISTRICT")
        province_entity = _best_entity(entities, "PROVINCE")
        issues: list[dict[str, str]] = []

        if ward_entity is None:
            return self._result(
                "REJECTED",
                None,
                None,
                None,
                None,
                [
                    {
                        "code": "LEGACY_WARD_MISSING",
                        "message": "legacy address has no WARD entity",
                    }
                ],
            )

        ward_candidates, ward_quality = self._lookup(
            ward_entity["text"],
            self.ward_exact,
            self.ward_folded,
        )
        district_quality: float | None = None
        province_quality: float | None = None

        if province_entity is not None:
            province_candidates, province_quality = self._lookup(
                province_entity["text"],
                self.province_exact,
                self.province_folded,
            )
            if not province_candidates:
                issues.append(
                    {
                        "code": "LEGACY_PROVINCE_NOT_FOUND",
                        "message": (
                            f"legacy province {province_entity['text']!r} "
                            "was not found"
                        ),
                    }
                )
                ward_candidates = []
            else:
                province_codes = {
                    record.legacy_province_code
                    for record in province_candidates
                }
                ward_candidates = [
                    record
                    for record in ward_candidates
                    if record.legacy_province_code in province_codes
                ]

        if district_entity is not None:
            district_candidates, district_quality = self._lookup(
                district_entity["text"],
                self.district_exact,
                self.district_folded,
            )
            if not district_candidates:
                issues.append(
                    {
                        "code": "LEGACY_DISTRICT_NOT_FOUND",
                        "message": (
                            f"legacy district {district_entity['text']!r} "
                            "was not found"
                        ),
                    }
                )
                ward_candidates = []
            else:
                district_codes = {
                    record.legacy_district_code
                    for record in district_candidates
                }
                ward_candidates = [
                    record
                    for record in ward_candidates
                    if record.legacy_district_code in district_codes
                ]

        if not ward_candidates:
            if not issues:
                issues.append(
                    {
                        "code": "LEGACY_WARD_NOT_FOUND",
                        "message": (
                            f"legacy ward {ward_entity['text']!r} was not found "
                            "under the supplied hierarchy"
                        ),
                    }
                )
            return self._result(
                "REJECTED",
                None,
                ward_quality,
                district_quality,
                province_quality,
                issues,
            )

        if len(ward_candidates) > 1:
            issues.append(
                {
                    "code": "LEGACY_WARD_AMBIGUOUS",
                    "message": (
                        f"legacy ward {ward_entity['text']!r} has "
                        f"{len(ward_candidates)} candidates"
                    ),
                }
            )
            return self._result(
                "AMBIGUOUS",
                None,
                ward_quality,
                district_quality,
                province_quality,
                issues,
                ward_candidates,
            )

        selected = ward_candidates[0]
        if self.has_mapping_catalog:
            return self._resolve_mapping_candidates(
                selected,
                ward_quality,
                district_quality,
                province_quality,
            )

        if not selected.current_ward_code:
            issues.append(
                {
                    "code": "LEGACY_MAPPING_NOT_AVAILABLE",
                    "message": (
                        "legacy unit was identified but has no direct current "
                        "mapping; note-based mapping is required"
                    ),
                }
            )
            return self._result(
                "REJECTED",
                selected,
                ward_quality,
                district_quality,
                province_quality,
                issues,
                ward_candidates,
            )

        issues.append(
            {
                "code": "LEGACY_ADDRESS_MAPPED",
                "message": "legacy unit was mapped through the official direct code row",
            }
        )
        return self._result(
            "WARNING",
            selected,
            ward_quality,
            district_quality,
            province_quality,
            issues,
            ward_candidates,
        )

    def _resolve_mapping_candidates(
        self,
        selected: LegacyComparison,
        ward_quality: float | None,
        district_quality: float | None,
        province_quality: float | None,
    ) -> dict[str, Any]:
        mappings = self.mappings_by_legacy_code.get(
            selected.legacy_ward_code, []
        )
        mapping_status, chosen = classify_legacy_mapping_candidates(mappings)
        if mapping_status == "REJECTED":
            return self._result(
                "REJECTED",
                selected,
                ward_quality,
                district_quality,
                province_quality,
                [
                    {
                        "code": "LEGACY_MAPPING_NOT_AVAILABLE",
                        "message": (
                            "legacy unit was identified but the official "
                            "comparison data provides no current mapping"
                        ),
                    }
                ],
            )

        if mapping_status == "WARNING":
            assert chosen is not None
            return self._result(
                "WARNING",
                selected,
                ward_quality,
                district_quality,
                province_quality,
                [
                    {
                        "code": "LEGACY_ADDRESS_MAPPED",
                        "message": (
                            "legacy unit was mapped through official "
                            f"{chosen.mapping_method.lower()} evidence"
                        ),
                    }
                ],
                current_mapping=chosen,
                mapping_candidates=mappings,
            )

        mapping_methods = {mapping.mapping_method for mapping in mappings}
        if mapping_methods == {"SOURCE_NOTE_HINT"}:
            ambiguous_issue = {
                "code": "LEGACY_MAPPING_HINT_ONLY",
                "message": (
                    "the legacy source row names a likely successor but "
                    "does not prove that the whole former area moved there"
                ),
            }
        else:
            ambiguous_issue = {
                "code": "LEGACY_MAPPING_AMBIGUOUS",
                "message": (
                    "legacy unit was split, only partly transferred, or "
                    "has multiple possible current successors"
                ),
            }
        return self._result(
            "AMBIGUOUS",
            selected,
            ward_quality,
            district_quality,
            province_quality,
            [ambiguous_issue],
            current_mapping=None,
            mapping_candidates=mappings,
            suppress_current=True,
        )

    @staticmethod
    def _result(
        status: str,
        selected: LegacyComparison | None,
        ward_quality: float | None,
        district_quality: float | None,
        province_quality: float | None,
        issues: list[dict[str, str]],
        candidates: list[LegacyComparison] | None = None,
        current_mapping: Any | None = None,
        mapping_candidates: list[Any] | None = None,
        suppress_current: bool = False,
    ) -> dict[str, Any]:
        qualities = [
            quality
            for quality in (ward_quality, district_quality, province_quality)
            if quality is not None
        ]
        current_ward_code = (
            current_mapping.current_ward_code
            if current_mapping is not None
            else selected.current_ward_code if selected is not None else ""
        )
        current_ward_name = (
            current_mapping.current_ward_name
            if current_mapping is not None
            else selected.current_ward_name if selected is not None else ""
        )
        current_province_code = (
            current_mapping.current_province_code
            if current_mapping is not None
            else selected.current_province_code if selected is not None else ""
        )
        current_province_name = (
            current_mapping.current_province_name
            if current_mapping is not None
            else selected.current_province_name if selected is not None else ""
        )
        match_method = (
            f"legacy_{current_mapping.mapping_method.lower()}"
            if current_mapping is not None
            else "legacy_direct_code"
        )
        current_available = bool(current_ward_code) and not suppress_current
        mapping_candidates = mapping_candidates or []
        candidate_wards_by_code = {
            mapping.current_ward_code: {
                "code": mapping.current_ward_code,
                "name": mapping.current_ward_name,
                "province_code": mapping.current_province_code,
                "score": ward_quality or 0.0,
                "mapping_method": mapping.mapping_method,
            }
            for mapping in mapping_candidates
            if mapping.current_ward_code
        }
        return {
            "status": status,
            "legacy": (
                {
                    "ward": {
                        "code": selected.legacy_ward_code,
                        "name": selected.legacy_ward_name,
                        "unit_type": _unit_type(selected.legacy_ward_name),
                    },
                    "district": {
                        "code": selected.legacy_district_code,
                        "name": selected.legacy_district_name,
                    },
                    "province": {
                        "code": selected.legacy_province_code,
                        "name": selected.legacy_province_name,
                    },
                }
                if selected
                else None
            ),
            "current": {
                "ward": (
                    {
                        "code": current_ward_code,
                        "name": current_ward_name,
                        "unit_type": _unit_type(current_ward_name),
                        "province_code": current_province_code,
                        "match_method": match_method,
                    }
                    if current_available
                    else None
                ),
                "province": (
                    {
                        "code": current_province_code,
                        "name": current_province_name,
                        "match_method": match_method,
                    }
                    if current_available
                    else None
                ),
            },
            "resolver_confidence": (
                0.0
                if status in {"AMBIGUOUS", "REJECTED"}
                else min(qualities) if qualities else 0.0
            ),
            "candidate_ward_codes": sorted(
                candidate_wards_by_code
                if mapping_candidates
                else {
                    candidate.current_ward_code
                    for candidate in (candidates or [])
                    if candidate.current_ward_code
                }
            ),
            "candidate_wards": (
                list(candidate_wards_by_code.values())
                if mapping_candidates
                else [
                    {
                        "code": candidate.current_ward_code,
                        "name": candidate.current_ward_name,
                        "province_code": candidate.current_province_code,
                        "score": ward_quality or 0.0,
                    }
                    for candidate in (candidates or [])
                    if candidate.current_ward_code
                ]
            ),
            "candidate_legacy_ward_codes": sorted(
                {
                    candidate.legacy_ward_code
                    for candidate in (candidates or [])
                }
            ),
            "issues": issues,
        }
