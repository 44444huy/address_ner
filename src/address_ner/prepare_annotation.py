"""Create reviewable character-span annotation tasks from raw addresses."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Configuration: edit these values before creating annotation tasks.
INPUT_PATH = Path("data/curated/addresses.txt")
CATALOG_PATH = Path("data/catalog/admin_units_2025.csv")
OUTPUT_PATH = Path("data/annotation/tasks.jsonl")
ID_PREFIX = "raw"
SOURCE_NAME = "real"


@dataclass(frozen=True)
class AdminUnit:
    province_code: str
    province_name: str
    ward_code: str
    ward_name: str
    unit_type: str
    short_name: str


@dataclass(frozen=True)
class TextComponent:
    start: int
    end: int
    text: str
    normalized: str


def normalize_for_match(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold()
    value = value.replace(".", "")
    return " ".join(value.split()).strip(" ,;")


def iter_components(text: str) -> Iterable[TextComponent]:
    """Yield non-empty comma/semicolon-delimited components with exact offsets."""
    for match in re.finditer(r"[^,;]+", text):
        raw_component = match.group(0)
        left_trimmed = raw_component.lstrip()
        component_text = left_trimmed.rstrip()
        if not component_text:
            continue
        start = match.start() + len(raw_component) - len(left_trimmed)
        end = start + len(component_text)
        yield TextComponent(
            start=start,
            end=end,
            text=component_text,
            normalized=normalize_for_match(component_text),
        )


def load_catalog(path: Path) -> list[AdminUnit]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [AdminUnit(**{field: row[field] for field in AdminUnit.__dataclass_fields__}) for row in csv.DictReader(file)]


def _province_aliases(unit: AdminUnit) -> set[str]:
    full_name = unit.province_name
    short_name = re.sub(r"^(Thành phố|Tỉnh)\s+", "", full_name, flags=re.IGNORECASE)
    aliases = {full_name, short_name}
    if full_name.casefold().startswith("thành phố "):
        aliases.update({f"TP {short_name}", f"TP. {short_name}"})
    return {normalize_for_match(alias) for alias in aliases}


def _ward_aliases(unit: AdminUnit) -> set[str]:
    aliases = {unit.ward_name, unit.short_name}
    if unit.unit_type == "PHƯỜNG":
        aliases.update({f"P {unit.short_name}", f"P. {unit.short_name}"})
    elif unit.unit_type == "XÃ":
        aliases.update({f"X {unit.short_name}", f"X. {unit.short_name}"})
    return {normalize_for_match(alias) for alias in aliases}


class CatalogMatcher:
    def __init__(self, units: list[AdminUnit]) -> None:
        self.provinces: dict[str, dict[str, AdminUnit]] = defaultdict(dict)
        self.wards: dict[str, dict[str, AdminUnit]] = defaultdict(dict)
        for unit in units:
            for alias in _province_aliases(unit):
                self.provinces[alias][unit.province_code] = unit
            for alias in _ward_aliases(unit):
                self.wards[alias][unit.ward_code] = unit

    def suggest(self, text: str) -> tuple[list[list[int | str]], list[dict]]:
        components = list(iter_components(text))
        selected_provinces: dict[int, AdminUnit] = {}

        for component_index, component in enumerate(components):
            candidates = list(self.provinces.get(component.normalized, {}).values())
            if len(candidates) == 1:
                selected_provinces[component_index] = candidates[0]

        province_codes = {
            unit.province_code for unit in selected_provinces.values()
        }
        labels: list[list[int | str]] = []
        suggestions: list[dict] = []

        for component_index, unit in selected_provinces.items():
            component = components[component_index]
            labels.append([component.start, component.end, "PROVINCE"])
            suggestions.append(
                {
                    "start": component.start,
                    "end": component.end,
                    "text": component.text,
                    "label": "PROVINCE",
                    "code": unit.province_code,
                    "method": "catalog_exact_component",
                }
            )

        for component_index, component in enumerate(components):
            if component_index in selected_provinces:
                continue
            candidates = list(self.wards.get(component.normalized, {}).values())
            if province_codes:
                candidates = [
                    unit for unit in candidates if unit.province_code in province_codes
                ]
            if len(candidates) != 1:
                continue

            unit = candidates[0]
            labels.append([component.start, component.end, "WARD"])
            suggestions.append(
                {
                    "start": component.start,
                    "end": component.end,
                    "text": component.text,
                    "label": "WARD",
                    "code": unit.ward_code,
                    "parent_code": unit.province_code,
                    "method": "catalog_exact_component",
                }
            )

        labels.sort(key=lambda label: (int(label[0]), int(label[1])))
        suggestions.sort(key=lambda suggestion: (suggestion["start"], suggestion["end"]))
        return labels, suggestions


def create_annotation_tasks(
    raw_addresses: Iterable[str],
    matcher: CatalogMatcher,
    id_prefix: str = "raw",
    source: str = "real",
) -> list[dict]:
    tasks: list[dict] = []
    for line_number, raw_address in enumerate(raw_addresses, start=1):
        text = unicodedata.normalize("NFC", raw_address.strip())
        if not text:
            continue
        labels, suggestions = matcher.suggest(text)
        tasks.append(
            {
                "id": f"{id_prefix}_{line_number:06d}",
                "text": text,
                "label": labels,
                "suggestions": suggestions,
                "annotation_status": "needs_review",
                "source": source,
            }
        )
    return tasks


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    units = load_catalog(CATALOG_PATH)
    matcher = CatalogMatcher(units)
    raw_addresses = INPUT_PATH.read_text(encoding="utf-8").splitlines()
    tasks = create_annotation_tasks(
        raw_addresses,
        matcher,
        id_prefix=ID_PREFIX,
        source=SOURCE_NAME,
    )
    write_jsonl(OUTPUT_PATH, tasks)

    suggested_spans = sum(len(task["label"]) for task in tasks)
    print(f"wrote {len(tasks)} task(s) to {OUTPUT_PATH}")
    print(f"suggested {suggested_spans} WARD/PROVINCE span(s); all tasks need review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
