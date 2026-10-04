"""Generate targeted synthetic cases for known parser failure modes."""

from __future__ import annotations

import random
import re
from pathlib import Path

from .generate_synthetic import (
    AddressBuilder,
    load_street_names,
    province_surface,
    random_house_number,
    remove_accents,
    select_balanced_units,
    ward_surface,
)
from .prepare_annotation import AdminUnit, load_catalog, write_jsonl
from .tokenize_dataset import load_jsonl


# Configuration: edit these values before generating hard cases.
CATALOG_PATH = Path("data/catalog/admin_units_2025.csv")
STREET_NAMES_PATH = Path("data/catalog/street_names.seed.txt")
BASE_DATASET_PATH = Path("data/synthetic/annotations.jsonl")
FORBIDDEN_DATASET_PATH = Path("data/golden/golden_cases.seed.jsonl")
NEGATIVE_COUNT = 64
LEGACY_COUNT = 160
FORMAT_COUNT = 160
RANDOM_SEED = 43
OUTPUT_PATH = Path("data/synthetic/annotations-v2.jsonl")


LEGACY_DISTRICTS = (
    ("Quận 1", "Thành phố Hồ Chí Minh"),
    ("Quận 3", "Thành phố Hồ Chí Minh"),
    ("Quận 5", "Thành phố Hồ Chí Minh"),
    ("Quận 7", "Thành phố Hồ Chí Minh"),
    ("Quận 10", "Thành phố Hồ Chí Minh"),
    ("Quận 12", "Thành phố Hồ Chí Minh"),
    ("Quận Cầu Giấy", "Thành phố Hà Nội"),
    ("Quận Đống Đa", "Thành phố Hà Nội"),
    ("Quận Hai Bà Trưng", "Thành phố Hà Nội"),
    ("Quận Hoàn Kiếm", "Thành phố Hà Nội"),
    ("Huyện Gia Lâm", "Thành phố Hà Nội"),
    ("Huyện Đông Anh", "Thành phố Hà Nội"),
    ("Quận Hải Châu", "Thành phố Đà Nẵng"),
    ("Quận Sơn Trà", "Thành phố Đà Nẵng"),
    ("Quận Ninh Kiều", "Thành phố Cần Thơ"),
    ("Quận Lê Chân", "Thành phố Hải Phòng"),
)

NEGATIVE_PATTERNS = (
    "giao hàng {time}",
    "vui lòng gọi trước {time}",
    "khách nhận hàng {time}",
    "chưa có thông tin người nhận {note}",
    "đơn hàng {note}",
    "liên hệ người nhận {time}",
    "nhận hàng tại cửa hàng {time}",
    "thông tin giao nhận {note}",
)
NEGATIVE_TIMES = (
    "vào buổi sáng",
    "sau giờ làm việc",
    "trước mười giờ",
    "trong giờ hành chính",
    "vào cuối tuần",
    "khi có thông báo",
    "sau khi gọi điện",
    "trước khi trời tối",
)
NEGATIVE_NOTES = (
    "đang cập nhật",
    "sẽ bổ sung sau",
    "không xác định",
    "cần kiểm tra lại",
    "chưa đầy đủ",
    "do khách cung cấp",
    "không có ghi chú",
    "đang chờ xác nhận",
)


def _record(
    record_id: str,
    group_id: str,
    builder: AddressBuilder,
    kind: str,
) -> dict:
    return {
        "id": record_id,
        "group_id": group_id,
        "text": builder.text,
        "label": builder.labels,
        "annotation_status": "approved",
        "source": "synthetic",
        "generation": {"kind": kind},
    }


def generate_negative_records(count: int, rng: random.Random) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    while len(records) < count:
        pattern_index = rng.randrange(len(NEGATIVE_PATTERNS))
        text = NEGATIVE_PATTERNS[pattern_index].format(
            time=rng.choice(NEGATIVE_TIMES), note=rng.choice(NEGATIVE_NOTES)
        )
        if text in seen:
            if len(seen) == len(NEGATIVE_PATTERNS) * max(
                len(NEGATIVE_TIMES), len(NEGATIVE_NOTES)
            ):
                raise ValueError("negative count exceeds unique template combinations")
            continue
        seen.add(text)
        builder = AddressBuilder()
        builder.add(text)
        records.append(
            _record(
                f"hard_negative_{len(records) + 1:06d}",
                f"hard_negative_pattern_{pattern_index}",
                builder,
                "negative",
            )
        )
    return records


def _legacy_surface(
    district: str, province: str, style: str
) -> tuple[str, str]:
    if style == "abbreviated":
        district = re.sub(r"^Quận\s+", "Q. ", district)
        district = re.sub(r"^Huyện\s+", "H. ", district)
        province = re.sub(r"^Thành phố Hồ Chí Minh$", "TP. HCM", province)
        province = re.sub(r"^Thành phố Hà Nội$", "TP. Hà Nội", province)
        province = re.sub(r"^Thành phố\s+", "TP. ", province)
    elif style == "ascii":
        district = remove_accents(district)
        province = remove_accents(province)
    return district, province


def generate_legacy_records(
    count: int, street_names: list[str], rng: random.Random
) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    while len(records) < count:
        district_index = rng.randrange(len(LEGACY_DISTRICTS))
        district, province = LEGACY_DISTRICTS[district_index]
        style = rng.choices(
            ["full", "abbreviated", "ascii"], weights=[45, 35, 20], k=1
        )[0]
        district, province = _legacy_surface(district, province, style)
        street = rng.choice(street_names)
        if style == "ascii":
            street = remove_accents(street)

        builder = AddressBuilder()
        if rng.random() < 0.8:
            builder.add(random_house_number(rng), "HOUSE_NUMBER")
            builder.add(" ")
        builder.add(street, "STREET")
        builder.add(rng.choice([", ", "; ", " - "]))
        builder.add(district, "DISTRICT")
        builder.add(rng.choice([", ", "; ", " - "]))
        builder.add(province, "PROVINCE")
        if builder.text.casefold() in seen:
            continue
        seen.add(builder.text.casefold())
        records.append(
            _record(
                f"hard_legacy_{len(records) + 1:06d}",
                f"hard_legacy_district_{district_index}",
                builder,
                "legacy_district",
            )
        )
    return records


def _complex_house_number(rng: random.Random) -> str:
    base = rng.randint(1, 299)
    return rng.choice(
        [
            f"{base}/{rng.randint(1, 30)}",
            f"{base}{rng.choice(['A', 'B', 'C'])}/{rng.randint(1, 30)}",
            f"{base}-{rng.randint(1, 20)}",
        ]
    )


def build_format_record(
    unit: AdminUnit,
    street: str,
    record_id: str,
    rng: random.Random,
) -> dict:
    abbreviated = rng.random() < 0.25
    builder = AddressBuilder()
    if rng.random() < 0.8:
        if rng.random() < 0.5:
            builder.add("Số ")
        builder.add(_complex_house_number(rng), "HOUSE_NUMBER")
        builder.add(", ")
    builder.add(rng.choice(["đường ", "Đường "]))
    builder.add(street, "STREET")
    builder.add(", ")
    builder.add(ward_surface(unit, abbreviated), "WARD")
    builder.add(", ")
    builder.add(province_surface(unit, abbreviated), "PROVINCE")
    return _record(
        record_id,
        f"hard_format_admin_{unit.ward_code}",
        builder,
        "street_prefix_complex_house",
    )


def generate_format_records(
    units: list[AdminUnit],
    street_names: list[str],
    count: int,
    rng: random.Random,
) -> list[dict]:
    selected_units = select_balanced_units(units, count, rng)
    return [
        build_format_record(
            unit,
            rng.choice(street_names),
            f"hard_format_{index:06d}",
            rng,
        )
        for index, unit in enumerate(selected_units, start=1)
    ]


def ensure_no_exact_leakage(records: list[dict], forbidden_records: list[dict]) -> None:
    forbidden = {record["text"].casefold() for record in forbidden_records}
    leaked = [record["id"] for record in records if record["text"].casefold() in forbidden]
    if leaked:
        raise ValueError(f"exact text leakage into forbidden dataset: {leaked[:10]}")


def main() -> int:
    rng = random.Random(RANDOM_SEED)
    street_names = load_street_names(STREET_NAMES_PATH)
    generated = [
        *generate_negative_records(NEGATIVE_COUNT, rng),
        *generate_legacy_records(LEGACY_COUNT, street_names, rng),
        *generate_format_records(
            load_catalog(CATALOG_PATH), street_names, FORMAT_COUNT, rng
        ),
    ]
    base = load_jsonl(BASE_DATASET_PATH)
    for record in base:
        record.setdefault("generation", {})["kind"] = "current"
    combined = [*base, *generated]
    ensure_no_exact_leakage(combined, load_jsonl(FORBIDDEN_DATASET_PATH))
    write_jsonl(OUTPUT_PATH, combined)

    print(f"base={len(base)}, generated={len(generated)}, combined={len(combined)}")
    print(f"- negative={NEGATIVE_COUNT}")
    print(f"- legacy={LEGACY_COUNT}")
    print(f"- format={FORMAT_COUNT}")
    print(f"wrote combined annotations to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
