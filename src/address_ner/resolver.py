"""Resolve parsed ward/province entities against the current admin catalog."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .prepare_annotation import AdminUnit, load_catalog


# Configuration: edit these values to try another resolver example.
WARD_TEXT: str | None = "P. Ba Dinh"
PROVINCE_TEXT: str | None = "TP Ha Noi"
WARD_CONFIDENCE: float | None = 0.92
PROVINCE_CONFIDENCE: float | None = 0.89
CATALOG_PATH = Path("data/catalog/admin_units_2025.csv")


MATCH_QUALITY = {
    "exact_alias": 1.0,
    "accent_folded_alias": 0.9,
    "inferred_from_ward": 0.95,
}


def normalize_admin_name(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold()
    value = value.replace(".", " ")
    value = re.sub(r"[,;|]+", " ", value)
    return " ".join(value.split())


def fold_accents(value: str) -> str:
    value = normalize_admin_name(value).replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )


def _initials(value: str) -> str:
    return "".join(word[0] for word in value.split() if word)


def province_aliases(unit: AdminUnit) -> set[str]:
    full_name = unit.province_name
    short_name = re.sub(
        r"^(Thành phố|Tỉnh)\s+", "", full_name, flags=re.IGNORECASE
    )
    aliases = {full_name, short_name}
    initials = _initials(short_name)
    if full_name.casefold().startswith("thành phố "):
        aliases.update(
            {
                f"TP {short_name}",
                f"TP. {short_name}",
                f"TP {initials}",
                f"TP. {initials}",
            }
        )
    elif full_name.casefold().startswith("tỉnh "):
        aliases.update({f"T {short_name}", f"T. {short_name}"})
    return {normalize_admin_name(alias) for alias in aliases}


def ward_aliases(unit: AdminUnit) -> set[str]:
    aliases = {unit.ward_name, unit.short_name}
    if unit.unit_type == "PHƯỜNG":
        aliases.update({f"P {unit.short_name}", f"P. {unit.short_name}"})
    elif unit.unit_type == "XÃ":
        aliases.update({f"X {unit.short_name}", f"X. {unit.short_name}"})
    elif unit.unit_type == "ĐẶC KHU":
        aliases.update({f"ĐK {unit.short_name}", f"ĐK. {unit.short_name}"})
    return {normalize_admin_name(alias) for alias in aliases}


def _register(
    index: dict[str, dict[str, AdminUnit]],
    key: str,
    identity: str,
    unit: AdminUnit,
) -> None:
    index[key][identity] = unit


def _confidence(entity: dict[str, Any] | None) -> float | None:
    if entity is None:
        return None
    value = entity.get("mean_confidence", entity.get("confidence"))
    return float(value) if value is not None else None


def _pick_entity(
    entities: Iterable[dict[str, Any]], entity_type: str
) -> tuple[dict[str, Any] | None, bool]:
    candidates = [entity for entity in entities if entity.get("type") == entity_type]
    if not candidates:
        return None, False
    return max(candidates, key=lambda entity: _confidence(entity) or 0.0), len(candidates) > 1


class AdminResolver:
    def __init__(
        self,
        units: list[AdminUnit],
        ward_fuzzy_threshold: float = 0.88,
        province_fuzzy_threshold: float = 0.86,
        fuzzy_margin: float = 0.03,
    ) -> None:
        self.ward_fuzzy_threshold = ward_fuzzy_threshold
        self.province_fuzzy_threshold = province_fuzzy_threshold
        self.fuzzy_margin = fuzzy_margin
        self.province_exact: dict[str, dict[str, AdminUnit]] = defaultdict(dict)
        self.province_folded: dict[str, dict[str, AdminUnit]] = defaultdict(dict)
        self.ward_exact: dict[str, dict[str, AdminUnit]] = defaultdict(dict)
        self.ward_folded: dict[str, dict[str, AdminUnit]] = defaultdict(dict)
        self.provinces_by_code: dict[str, AdminUnit] = {}
        self._fuzzy_rank_cache: dict[
            tuple[str, str, str | None], list[tuple[AdminUnit, float]]
        ] = {}

        for unit in units:
            self.provinces_by_code[unit.province_code] = unit
            for alias in province_aliases(unit):
                _register(
                    self.province_exact,
                    alias,
                    unit.province_code,
                    unit,
                )
                _register(
                    self.province_folded,
                    fold_accents(alias),
                    unit.province_code,
                    unit,
                )
            for alias in ward_aliases(unit):
                _register(self.ward_exact, alias, unit.ward_code, unit)
                _register(
                    self.ward_folded,
                    fold_accents(alias),
                    unit.ward_code,
                    unit,
                )

    @staticmethod
    def _lookup(
        text: str,
        exact_index: dict[str, dict[str, AdminUnit]],
        folded_index: dict[str, dict[str, AdminUnit]],
    ) -> tuple[list[AdminUnit], str | None]:
        exact = list(exact_index.get(normalize_admin_name(text), {}).values())
        if exact:
            return exact, "exact_alias"
        folded = list(folded_index.get(fold_accents(text), {}).values())
        if folded:
            return folded, "accent_folded_alias"
        return [], None

    def _fuzzy_lookup(
        self,
        text: str,
        folded_index: dict[str, dict[str, AdminUnit]],
        identity_field: str,
        threshold: float,
        province_code: str | None = None,
    ) -> tuple[list[AdminUnit], float | None, list[tuple[AdminUnit, float]]]:
        query = fold_accents(text)
        cache_key = (query, identity_field, province_code)
        ranked = self._fuzzy_rank_cache.get(cache_key)
        if ranked is None:
            scores: dict[str, float] = {}
            units_by_identity: dict[str, AdminUnit] = {}
            for alias, units in folded_index.items():
                score = SequenceMatcher(None, query, alias).ratio()
                for unit in units.values():
                    if (
                        province_code is not None
                        and unit.province_code != province_code
                    ):
                        continue
                    identity = str(getattr(unit, identity_field))
                    if score > scores.get(identity, -1.0):
                        scores[identity] = score
                        units_by_identity[identity] = unit

            ranked = sorted(
                (
                    (units_by_identity[identity], score)
                    for identity, score in scores.items()
                ),
                key=lambda item: (-item[1], getattr(item[0], identity_field)),
            )
            self._fuzzy_rank_cache[cache_key] = ranked
        suggestions = ranked[:3]
        if not ranked or ranked[0][1] < threshold:
            return [], ranked[0][1] if ranked else None, suggestions

        top_score = ranked[0][1]
        if len(ranked) > 1 and top_score - ranked[1][1] < self.fuzzy_margin:
            ambiguous = [
                unit for unit, score in ranked if top_score - score < self.fuzzy_margin
            ]
            return ambiguous, top_score, suggestions
        return [ranked[0][0]], top_score, suggestions

    def resolve(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[dict[str, str]] = []
        ward_entity, multiple_wards = _pick_entity(entities, "WARD")
        province_entity, multiple_provinces = _pick_entity(entities, "PROVINCE")
        selected_entities = [
            entity for entity in (ward_entity, province_entity) if entity is not None
        ]

        if multiple_wards:
            issues.append(
                {
                    "code": "MULTIPLE_WARD_ENTITIES",
                    "message": "multiple WARD entities were predicted; highest confidence was used",
                }
            )
        if multiple_provinces:
            issues.append(
                {
                    "code": "MULTIPLE_PROVINCE_ENTITIES",
                    "message": "multiple PROVINCE entities were predicted; highest confidence was used",
                }
            )
        if ward_entity is None and province_entity is None:
            return self._result(
                "REJECTED",
                ward_entity,
                province_entity,
                None,
                None,
                None,
                None,
                [{"code": "NO_ADMIN_ENTITIES", "message": "no WARD or PROVINCE entity"}],
                selected_entities,
            )

        province_unit: AdminUnit | None = None
        province_method: str | None = None
        province_quality: float | None = None
        if province_entity is not None:
            province_candidates, province_method = self._lookup(
                province_entity["text"], self.province_exact, self.province_folded
            )
            if not province_candidates:
                (
                    province_candidates,
                    province_quality,
                    _province_suggestions,
                ) = self._fuzzy_lookup(
                    province_entity["text"],
                    self.province_folded,
                    "province_code",
                    self.province_fuzzy_threshold,
                )
                if province_candidates:
                    province_method = "fuzzy_alias"
            if len(province_candidates) == 1:
                province_unit = province_candidates[0]
                if province_method != "fuzzy_alias":
                    province_quality = MATCH_QUALITY[province_method]
                else:
                    issues.append(
                        {
                            "code": "PROVINCE_FUZZY_MATCH",
                            "message": f"province was fuzzy-matched with score {province_quality:.3f}",
                        }
                    )
            elif not province_candidates:
                issues.append(
                    {
                        "code": "PROVINCE_NOT_FOUND",
                        "message": f"province {province_entity['text']!r} was not found",
                    }
                )
            else:
                issues.append(
                    {
                        "code": "PROVINCE_AMBIGUOUS",
                        "message": f"province {province_entity['text']!r} has multiple candidates",
                    }
                )

        ward_unit: AdminUnit | None = None
        ward_method: str | None = None
        ward_quality: float | None = None
        ward_candidates: list[AdminUnit] = []
        ward_suggestions: list[tuple[AdminUnit, float]] = []
        if ward_entity is not None:
            all_ward_candidates, ward_method = self._lookup(
                ward_entity["text"], self.ward_exact, self.ward_folded
            )
            ward_candidates = all_ward_candidates
            if province_unit is not None:
                ward_candidates = [
                    unit
                    for unit in all_ward_candidates
                    if unit.province_code == province_unit.province_code
                ]
            if not all_ward_candidates:
                (
                    ward_candidates,
                    ward_quality,
                    ward_suggestions,
                ) = self._fuzzy_lookup(
                    ward_entity["text"],
                    self.ward_folded,
                    "ward_code",
                    self.ward_fuzzy_threshold,
                    province_unit.province_code if province_unit else None,
                )
                ward_method = "fuzzy_alias" if ward_candidates else None

                if not ward_candidates and province_unit is not None:
                    (
                        global_candidates,
                        _global_score,
                        global_suggestions,
                    ) = self._fuzzy_lookup(
                        ward_entity["text"],
                        self.ward_folded,
                        "ward_code",
                        self.ward_fuzzy_threshold,
                    )
                    if global_candidates:
                        all_ward_candidates = global_candidates
                        ward_suggestions = global_suggestions
            if len(ward_candidates) == 1:
                ward_unit = ward_candidates[0]
                if ward_method != "fuzzy_alias":
                    ward_quality = MATCH_QUALITY[ward_method]
                else:
                    issues.append(
                        {
                            "code": "WARD_FUZZY_MATCH",
                            "message": f"ward was fuzzy-matched with score {ward_quality:.3f}",
                        }
                    )
            elif not ward_candidates and all_ward_candidates and province_unit is not None:
                issues.append(
                    {
                        "code": "WARD_PROVINCE_CONFLICT",
                        "message": "ward candidates do not belong to the resolved province",
                    }
                )
            elif not ward_candidates:
                issues.append(
                    {
                        "code": "WARD_NOT_FOUND",
                        "message": f"ward {ward_entity['text']!r} was not found",
                    }
                )
            else:
                issues.append(
                    {
                        "code": "WARD_AMBIGUOUS",
                        "message": f"ward {ward_entity['text']!r} has {len(ward_candidates)} candidates",
                    }
                )

        if ward_unit is not None and province_entity is None:
            province_unit = self.provinces_by_code[ward_unit.province_code]
            province_method = "inferred_from_ward"
            province_quality = MATCH_QUALITY[province_method]
            issues.append(
                {
                    "code": "PROVINCE_INFERRED_FROM_WARD",
                    "message": "province was inferred from the unique ward candidate",
                }
            )

        issue_codes = {issue["code"] for issue in issues}
        if issue_codes & {"WARD_AMBIGUOUS", "PROVINCE_AMBIGUOUS"}:
            status = "AMBIGUOUS"
        elif issue_codes & {
            "NO_ADMIN_ENTITIES",
            "PROVINCE_NOT_FOUND",
            "WARD_NOT_FOUND",
            "WARD_PROVINCE_CONFLICT",
        }:
            status = "REJECTED"
        elif ward_unit is None or province_unit is None or issues:
            status = "WARNING"
        else:
            status = "PASS"

        return self._result(
            status,
            ward_entity,
            province_entity,
            ward_unit,
            province_unit,
            ward_method,
            province_method,
            issues,
            selected_entities,
            ward_quality,
            province_quality,
            ward_suggestions
            or [
                (unit, ward_quality or MATCH_QUALITY.get(ward_method or "", 0.0))
                for unit in ward_candidates
            ],
        )

    @staticmethod
    def _result(
        status: str,
        ward_entity: dict[str, Any] | None,
        province_entity: dict[str, Any] | None,
        ward_unit: AdminUnit | None,
        province_unit: AdminUnit | None,
        ward_method: str | None,
        province_method: str | None,
        issues: list[dict[str, str]],
        selected_entities: list[dict[str, Any]],
        ward_quality: float | None = None,
        province_quality: float | None = None,
        ward_candidates: list[tuple[AdminUnit, float]] | None = None,
    ) -> dict[str, Any]:
        parser_confidences = [
            confidence
            for confidence in (_confidence(entity) for entity in selected_entities)
            if confidence is not None
        ]
        match_qualities = [
            quality for quality in (ward_quality, province_quality) if quality is not None
        ]
        return {
            "status": status,
            "input": {
                "ward": ward_entity["text"] if ward_entity else None,
                "province": province_entity["text"] if province_entity else None,
            },
            "current": {
                "ward": (
                    {
                        "code": ward_unit.ward_code,
                        "name": ward_unit.ward_name,
                        "unit_type": ward_unit.unit_type,
                        "province_code": ward_unit.province_code,
                        "match_method": ward_method,
                    }
                    if ward_unit
                    else None
                ),
                "province": (
                    {
                        "code": province_unit.province_code,
                        "name": province_unit.province_name,
                        "match_method": province_method,
                    }
                    if province_unit
                    else None
                ),
            },
            "parser_confidence": (
                sum(parser_confidences) / len(parser_confidences)
                if parser_confidences
                else None
            ),
            "resolver_confidence": (
                0.0
                if status in {"AMBIGUOUS", "REJECTED"}
                else min(match_qualities) if match_qualities else 0.0
            ),
            "candidate_ward_codes": sorted(
                unit.ward_code for unit, _score in (ward_candidates or [])
            ),
            "candidate_wards": [
                {
                    "code": unit.ward_code,
                    "name": unit.ward_name,
                    "province_code": unit.province_code,
                    "score": score,
                }
                for unit, score in (ward_candidates or [])
            ],
            "issues": issues,
        }


def main() -> int:
    entities: list[dict[str, Any]] = []
    if WARD_TEXT:
        entities.append(
            {"type": "WARD", "text": WARD_TEXT, "confidence": WARD_CONFIDENCE}
        )
    if PROVINCE_TEXT:
        entities.append(
            {
                "type": "PROVINCE",
                "text": PROVINCE_TEXT,
                "confidence": PROVINCE_CONFIDENCE,
            }
        )
    result = AdminResolver(load_catalog(CATALOG_PATH)).resolve(entities)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
