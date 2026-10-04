"""Generate deterministic synthetic training annotations from the admin catalog."""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .prepare_annotation import AdminUnit, load_catalog, write_jsonl


# Configuration: edit these values before generating synthetic records.
CATALOG_PATH = Path("data/catalog/admin_units_2025.csv")
STREET_NAMES_PATH = Path("data/catalog/street_names.seed.txt")
RECORD_COUNT = 1000
VARIANTS_PER_UNIT = 2
RANDOM_SEED = 42
OUTPUT_PATH = Path("data/synthetic/annotations.jsonl")


@dataclass
class AddressBuilder:
    parts: list[str] = field(default_factory=list)
    labels: list[list[int | str]] = field(default_factory=list)
    length: int = 0

    def add(self, text: str, entity_type: str | None = None) -> None:
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        if entity_type is not None:
            self.labels.append([start, self.length, entity_type])

    @property
    def text(self) -> str:
        return "".join(self.parts)


def remove_accents(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", value)
    return unicodedata.normalize(
        "NFC", "".join(character for character in decomposed if unicodedata.category(character) != "Mn")
    )


def ward_surface(unit: AdminUnit, abbreviated: bool) -> str:
    if not abbreviated:
        return unit.ward_name
    if unit.unit_type == "PHƯỜNG":
        return f"P. {unit.short_name}"
    if unit.unit_type == "XÃ":
        return f"X. {unit.short_name}"
    return unit.ward_name


def province_surface(unit: AdminUnit, abbreviated: bool) -> str:
    if abbreviated and unit.province_name.casefold().startswith("thành phố "):
        short_name = re.sub(
            r"^Thành phố\s+", "", unit.province_name, flags=re.IGNORECASE
        )
        return f"TP. {short_name}"
    return unit.province_name


def random_house_number(rng: random.Random) -> str:
    base = str(rng.randint(1, 299))
    style = rng.choices(["simple", "suffix", "slash"], weights=[7, 2, 1], k=1)[0]
    if style == "suffix":
        return f"{base}{rng.choice(['A', 'B', 'C'])}"
    if style == "slash":
        return f"{base}/{rng.randint(1, 30)}"
    return base


def build_synthetic_record(
    unit: AdminUnit,
    street_name: str,
    record_id: str,
    rng: random.Random,
) -> dict:
    style = rng.choices(
        ["full", "abbreviated", "ascii", "uppercase"],
        weights=[60, 20, 15, 5],
        k=1,
    )[0]
    abbreviated = style == "abbreviated"
    include_street = rng.random() < 0.88
    include_house = include_street and rng.random() < 0.72
    separator = rng.choices([", ", "; ", " - ", " "], weights=[70, 10, 10, 10], k=1)[0]
    if separator == " " and style in {"abbreviated", "ascii"}:
        separator = ", "

    components: list[tuple[str, str]] = []
    if include_house:
        components.append((random_house_number(rng), "HOUSE_NUMBER"))
    if include_street:
        components.append((street_name, "STREET"))
    components.append((ward_surface(unit, abbreviated), "WARD"))
    components.append((province_surface(unit, abbreviated), "PROVINCE"))

    if style == "ascii":
        components = [(remove_accents(text), entity_type) for text, entity_type in components]
    elif style == "uppercase":
        components = [(text.upper(), entity_type) for text, entity_type in components]

    builder = AddressBuilder()
    first_component = 0
    if include_house and include_street:
        house_text, house_type = components[0]
        street_text, street_type = components[1]
        builder.add(house_text, house_type)
        builder.add(" ")
        builder.add(street_text, street_type)
        first_component = 2
    elif include_street:
        street_text, street_type = components[0]
        builder.add(street_text, street_type)
        first_component = 1

    for component_index, (text, entity_type) in enumerate(
        components[first_component:], start=first_component
    ):
        if builder.length:
            builder.add(separator)
        builder.add(text, entity_type)

    return {
        "id": record_id,
        "group_id": f"synthetic_admin_{unit.ward_code}",
        "text": builder.text,
        "label": builder.labels,
        "annotation_status": "approved",
        "source": "synthetic",
        "generation": {
            "style": style,
            "separator": separator,
            "province_code": unit.province_code,
            "ward_code": unit.ward_code,
        },
    }


def select_balanced_units(
    units: list[AdminUnit], count: int, rng: random.Random
) -> list[AdminUnit]:
    """Round-robin provinces so small provinces are represented too."""
    by_province: dict[str, list[AdminUnit]] = defaultdict(list)
    for unit in units:
        by_province[unit.province_code].append(unit)
    for province_units in by_province.values():
        rng.shuffle(province_units)

    province_codes = sorted(by_province)
    rng.shuffle(province_codes)
    selected: list[AdminUnit] = []
    offset = 0
    while len(selected) < count:
        added = False
        for province_code in province_codes:
            province_units = by_province[province_code]
            if offset < len(province_units):
                selected.append(province_units[offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    return selected


def generate_records(
    units: list[AdminUnit],
    street_names: list[str],
    count: int,
    variants_per_unit: int,
    seed: int,
) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    if variants_per_unit <= 0:
        raise ValueError("variants_per_unit must be positive")
    if not units:
        raise ValueError("admin catalog is empty")
    if not street_names:
        raise ValueError("street name list is empty")

    group_count = math.ceil(count / variants_per_unit)
    if group_count > len(units):
        raise ValueError(
            f"count requires {group_count} admin units, but catalog has {len(units)}"
        )

    rng = random.Random(seed)
    selected_units = select_balanced_units(units, group_count, rng)
    records: list[dict] = []
    seen_texts: set[str] = set()

    for unit in selected_units:
        for _ in range(variants_per_unit):
            if len(records) == count:
                break
            for _attempt in range(100):
                record = build_synthetic_record(
                    unit,
                    rng.choice(street_names),
                    f"synthetic_{len(records) + 1:06d}",
                    rng,
                )
                key = record["text"].casefold()
                if key not in seen_texts:
                    seen_texts.add(key)
                    records.append(record)
                    break
            else:
                raise RuntimeError(f"could not create a unique variant for {unit.ward_code}")

    rng.shuffle(records)
    for index, record in enumerate(records, start=1):
        record["id"] = f"synthetic_{index:06d}"
    return records


def load_street_names(path: Path) -> list[str]:
    return [
        unicodedata.normalize("NFC", line.strip())
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    records = generate_records(
        load_catalog(CATALOG_PATH),
        load_street_names(STREET_NAMES_PATH),
        count=RECORD_COUNT,
        variants_per_unit=VARIANTS_PER_UNIT,
        seed=RANDOM_SEED,
    )
    write_jsonl(OUTPUT_PATH, records)

    style_counts: dict[str, int] = defaultdict(int)
    for record in records:
        style_counts[record["generation"]["style"]] += 1
    print(f"wrote {len(records)} synthetic annotation(s) to {OUTPUT_PATH}")
    print(f"groups={len({record['group_id'] for record in records})}")
    for style, style_count in sorted(style_counts.items()):
        print(f"- {style}: {style_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
