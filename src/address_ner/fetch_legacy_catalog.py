"""Fetch official pre/post-reform administrative comparison catalogs."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

from .fetch_admin_catalog import (
    GRID_CALLBACK_ID,
    SOURCE_URL,
    _decode_response,
    build_page_callback,
    parse_form_inputs,
    parse_grid_rows,
)


SOURCE_DATE = "30/06/2025"
TARGET_DATE = "30/09/2025"
REFORM_DATE = datetime(2025, 7, 1, tzinfo=timezone.utc)
WARD_OUTPUT_PATH = Path("data/catalog/ward_comparison_2025.csv")
PROVINCE_OUTPUT_PATH = Path("data/catalog/province_comparison_2025.csv")
REPORT_PATH = Path("data/catalog/legacy_catalog.report.json")


def build_custom_callback(argument: str = "") -> str:
    command = "CUSTOMCALLBACK"
    serialized = f"{len(command)}|{command}{len(argument)}|{argument}"
    return f"GB|{len(serialized)};{serialized};"


def _milliseconds(date_text: str) -> str:
    date = datetime.strptime(date_text, "%d/%m/%Y").replace(tzinfo=timezone.utc)
    return str(int(date.timestamp() * 1000))


def _form_key(values: dict[str, str], suffix: str) -> str:
    return next(key for key in values if key.endswith(suffix))


def _is_after_reform(date_text: str) -> bool:
    date = datetime.strptime(date_text, "%d/%m/%Y").replace(tzinfo=timezone.utc)
    return date >= REFORM_DATE


def _prepare_comparison_form(
    initial_html: str,
    grid_name: str,
    level_label: str,
    source_date: str,
    target_date: str,
) -> dict[str, str]:
    values = parse_form_inputs(initial_html)
    values[_form_key(values, "cmbCap")] = level_label
    values[_form_key(values, "txtNgay")] = source_date
    values[_form_key(values, "txtNgayDC")] = target_date
    values[_form_key(values, "txtBoQH")] = (
        "1" if _is_after_reform(source_date) else "0"
    )
    values[_form_key(values, "txtBoQH_DC")] = (
        "1" if _is_after_reform(target_date) else "0"
    )
    values["ctl00_PlaceHolderMain_txtNgay_Raw"] = _milliseconds(source_date)
    values["ctl00_PlaceHolderMain_txtNgayDC_Raw"] = _milliseconds(target_date)
    values["__CALLBACKID"] = GRID_CALLBACK_ID.replace("gridXa", grid_name)
    values["__CALLBACKPARAM"] = build_custom_callback()
    return values


def _page_count(callback_html: str) -> int:
    match = re.search(r"Page 1 of (\d+) \([\d,\.]+ items\)", callback_html)
    return int(match.group(1)) if match else 1


def fetch_comparison_rows(
    grid_name: str,
    level_label: str,
    source_date: str = SOURCE_DATE,
    target_date: str = TARGET_DATE,
    source_url: str = SOURCE_URL,
) -> list[list[str]]:
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    initial_html = _decode_response(opener.open(source_url, timeout=60))
    form_values = _prepare_comparison_form(
        initial_html,
        grid_name,
        level_label,
        source_date,
        target_date,
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "User-Agent": "address-ner-legacy-catalog-importer/1.0",
    }

    def callback() -> str:
        request = Request(
            source_url,
            data=urlencode(form_values).encode("utf-8"),
            headers=headers,
        )
        return _decode_response(opener.open(request, timeout=60))

    first_page = callback()
    pages = [first_page]
    form_values.update(parse_form_inputs(first_page))
    for page_index in range(1, _page_count(first_page)):
        form_values["__CALLBACKPARAM"] = build_page_callback(page_index)
        page = callback()
        pages.append(page)
        form_values.update(parse_form_inputs(page))

    return [
        row
        for page in pages
        for row in parse_grid_rows(page, grid_name=grid_name)
    ]


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def normalize_ward_comparison_rows(
    rows: list[list[str]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 15:
            raise ValueError(
                f"ward comparison row {row_number} has {len(row)} columns; "
                f"expected 15: {row!r}"
            )
        records.append(
            {
                "legacy_province_code": row[0],
                "legacy_province_name": _normalized(row[1]),
                "legacy_district_code": row[2],
                "legacy_district_name": _normalized(row[3]),
                "legacy_ward_code": row[4],
                "legacy_ward_name": _normalized(row[5]),
                "legacy_effective_from": row[7],
                "current_ward_code": row[9],
                "current_ward_name": _normalized(row[8]),
                "current_province_code": row[13],
                "current_province_name": _normalized(row[12]),
                "current_effective_from": row[11],
                "note": _normalized(row[14]),
                "source_url": SOURCE_URL,
            }
        )
    return records


def normalize_province_comparison_rows(
    rows: list[list[str]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 9:
            raise ValueError(
                f"province comparison row {row_number} has {len(row)} columns; "
                f"expected 9: {row!r}"
            )
        records.append(
            {
                "legacy_province_code": row[0],
                "legacy_province_name": _normalized(row[1]),
                "legacy_effective_from": row[3],
                "current_province_code": row[5],
                "current_province_name": _normalized(row[4]),
                "current_effective_from": row[7],
                "note": _normalized(row[8]),
                "source_url": SOURCE_URL,
            }
        )
    return records


def build_report(
    ward_records: list[dict[str, str]],
    province_records: list[dict[str, str]],
) -> dict[str, Any]:
    legacy_wards = [row for row in ward_records if row["legacy_ward_code"]]
    current_wards = [row for row in ward_records if row["current_ward_code"]]
    direct = [
        row
        for row in ward_records
        if row["legacy_ward_code"] and row["current_ward_code"]
    ]
    return {
        "source_date": SOURCE_DATE,
        "target_date": TARGET_DATE,
        "ward_comparison_rows": len(ward_records),
        "legacy_ward_rows": len(legacy_wards),
        "current_ward_rows": len(current_wards),
        "direct_ward_mappings": len(direct),
        "legacy_only_rows": sum(
            bool(row["legacy_ward_code"]) and not row["current_ward_code"]
            for row in ward_records
        ),
        "current_only_rows": sum(
            bool(row["current_ward_code"]) and not row["legacy_ward_code"]
            for row in ward_records
        ),
        "legacy_province_rows": sum(
            bool(row["legacy_province_code"]) for row in province_records
        ),
        "province_comparison_rows": len(province_records),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty catalog to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ward_records = normalize_ward_comparison_rows(
        fetch_comparison_rows(
            "gridXa",
            "Xã",
            SOURCE_DATE,
            TARGET_DATE,
        )
    )
    province_records = normalize_province_comparison_rows(
        fetch_comparison_rows(
            "gridTinh",
            "Tỉnh",
            SOURCE_DATE,
            TARGET_DATE,
        )
    )
    report = build_report(ward_records, province_records)
    write_csv(WARD_OUTPUT_PATH, ward_records)
    write_csv(PROVINCE_OUTPUT_PATH, province_records)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
