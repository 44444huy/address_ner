"""Fetch and normalize Vietnam's current administrative-unit catalog."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener


# Configuration: edit these paths before downloading the catalog.
OUTPUT_PATH = Path("data/catalog/admin_units_2025.csv")
REPORT_PATH = Path("data/catalog/admin_units_2025.report.json")


SOURCE_URL = "https://danhmuchanhchinh.nso.gov.vn/Doi_Chieu_Moi.aspx"
GRID_CALLBACK_ID = "ctl00$PlaceHolderMain$gridXa"
EXPECTED_PROVINCES = 34
EXPECTED_WARDS = 3321


class FormInputParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() != "input":
            return
        attributes = dict(attrs)
        name = attributes.get("name")
        input_type = (attributes.get("type") or "text").lower()
        if (
            not name
            or "disabled" in attributes
            or input_type in {"button", "file", "image", "reset", "submit"}
        ):
            return
        if input_type in {"checkbox", "radio"} and "checked" not in attributes:
            return
        self.values[name] = attributes.get("value") or ""


class GridRowParser(HTMLParser):
    """Extract visible cells from DevExpress ward data rows."""

    def __init__(self, grid_name: str = "gridXa") -> None:
        super().__init__()
        self.row_id_marker = f"{grid_name}_DXDataRow"
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag.lower() == "tr" and self.row_id_marker in (
            attributes.get("id") or ""
        ):
            self._row = []
        elif tag.lower() == "td" and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "td" and self._row is not None and self._cell_parts is not None:
            raw_value = "".join(self._cell_parts)
            for escaped_whitespace in ("\\r", "\\n", "\\t"):
                raw_value = raw_value.replace(escaped_whitespace, " ")
            value = " ".join(raw_value.split())
            self._row.append(value)
            self._cell_parts = None
        elif tag.lower() == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
            self._cell_parts = None


def parse_form_inputs(html: str) -> dict[str, str]:
    parser = FormInputParser()
    parser.feed(html)
    return parser.values


def parse_grid_rows(html: str, grid_name: str = "gridXa") -> list[list[str]]:
    parser = GridRowParser(grid_name)
    parser.feed(html)
    return parser.rows


def build_page_callback(page_index: int) -> str:
    """Build the DevExpress callback argument; page_index is zero-based."""
    command = "PAGERONCLICK"
    target = f"PN{page_index}"
    serialized = f"{len(command)}|{command}{len(target)}|{target}"
    return f"GB|{len(serialized)};{serialized};"


def _decode_response(response) -> str:
    content_type = response.headers.get_content_charset() or "utf-8"
    return response.read().decode(content_type, errors="replace")


def fetch_grid_pages(source_url: str = SOURCE_URL) -> list[str]:
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    initial_response = opener.open(source_url, timeout=60)
    initial_html = _decode_response(initial_response)
    pages = [initial_html]
    form_values = parse_form_inputs(initial_html)

    for page_index in range(1, 5):
        payload = {
            **form_values,
            "__CALLBACKID": GRID_CALLBACK_ID,
            "__CALLBACKPARAM": build_page_callback(page_index),
        }
        request = Request(
            source_url,
            data=urlencode(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "User-Agent": "address-ner-catalog-importer/1.0",
            },
        )
        pages.append(_decode_response(opener.open(request, timeout=60)))

    return pages


def _unit_type_and_short_name(full_name: str) -> tuple[str, str]:
    full_name = unicodedata.normalize("NFC", full_name)
    match = re.match(r"^(Phường|Xã|Đặc khu)\s+(.+)$", full_name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unsupported ward name format: {full_name!r}")
    return match.group(1).upper(), match.group(2)


def normalize_rows(rows: list[list[str]]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 13:
            raise ValueError(
                f"row {row_number} has {len(row)} columns; expected 13: {row!r}"
            )

        ward_name = unicodedata.normalize("NFC", row[6])
        ward_code = row[7]
        province_name = unicodedata.normalize("NFC", row[10])
        province_code = row[11]
        unit_type, short_name = _unit_type_and_short_name(ward_name)
        units.append(
            {
                "province_code": province_code,
                "province_name": province_name,
                "ward_code": ward_code,
                "ward_name": ward_name,
                "unit_type": unit_type,
                "short_name": short_name,
                "effective_from": row[9],
                "source_url": SOURCE_URL,
            }
        )
    return units


def validate_catalog(units: list[dict[str, str]]) -> dict[str, Any]:
    province_codes = {unit["province_code"] for unit in units}
    ward_codes = [unit["ward_code"] for unit in units]
    duplicate_ward_codes = sorted(
        code for code, count in Counter(ward_codes).items() if count > 1
    )

    errors: list[str] = []
    if len(units) != EXPECTED_WARDS:
        errors.append(f"expected {EXPECTED_WARDS} wards, found {len(units)}")
    if len(province_codes) != EXPECTED_PROVINCES:
        errors.append(
            f"expected {EXPECTED_PROVINCES} provinces, found {len(province_codes)}"
        )
    if duplicate_ward_codes:
        errors.append(f"duplicate ward codes: {duplicate_ward_codes[:10]}")

    invalid_province_codes = sorted(
        code for code in province_codes if not re.fullmatch(r"\d{2}", code)
    )
    invalid_ward_codes = sorted(
        code for code in ward_codes if not re.fullmatch(r"\d{5}", code)
    )
    if invalid_province_codes:
        errors.append(f"invalid province codes: {invalid_province_codes[:10]}")
    if invalid_ward_codes:
        errors.append(f"invalid ward codes: {invalid_ward_codes[:10]}")

    return {
        "valid": not errors,
        "errors": errors,
        "ward_count": len(units),
        "province_count": len(province_codes),
        "unit_type_counts": dict(sorted(Counter(unit["unit_type"] for unit in units).items())),
    }


def write_catalog(path: Path, units: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(units[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(units)


def main() -> int:
    pages = fetch_grid_pages()
    rows = [row for page in pages for row in parse_grid_rows(page)]
    units = normalize_rows(rows)
    report = validate_catalog(units)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not report["valid"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    write_catalog(OUTPUT_PATH, units)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
