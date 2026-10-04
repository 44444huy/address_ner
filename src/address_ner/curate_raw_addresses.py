"""Clean, deduplicate and profile raw addresses before annotation."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


# Configuration: edit these values before running this module.
INPUT_PATH = Path("data/raw/addresses.example.txt")
CSV_ADDRESS_COLUMN: str | None = None
SOURCE_NAME: str | None = None
OUTPUT_PATH = Path("data/curated/addresses.jsonl")
TEXT_OUTPUT_PATH = Path("data/curated/addresses.txt")
REJECTED_OUTPUT_PATH = Path("data/curated/rejected.jsonl")
REPORT_PATH = Path("data/curated/report.json")


DEFAULT_ADDRESS_COLUMNS = (
    "address",
    "raw_address",
    "full_address",
    "dia_chi",
    "địa_chỉ",
)
ZERO_WIDTH_CHARACTERS = "\u200b\u200c\u200d\ufeff"
EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){9,10}(?!\d)"
)
HOUSE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:số\s+)?\d+[a-z]?(?:\s*[/.-]\s*\d+[a-z]?)*\b",
    flags=re.IGNORECASE,
)
ABBREVIATION_PATTERN = re.compile(
    r"(?<!\w)(?:p|q|tp|t|x|h|tx|tt)\.(?=\s|\w)",
    flags=re.IGNORECASE,
)
WARD_MARKER_PATTERN = re.compile(
    r"\b(?:phường|xã|đặc\s+khu)\b|(?<!\w)[px]\.",
    flags=re.IGNORECASE,
)
DISTRICT_MARKER_PATTERN = re.compile(
    r"\b(?:quận|huyện|thị\s+xã)\b|(?<!\w)q\.",
    flags=re.IGNORECASE,
)
PROVINCE_MARKER_PATTERN = re.compile(
    r"\b(?:tỉnh|thành\s+phố)\b|(?<!\w)tp\.",
    flags=re.IGNORECASE,
)


def normalize_address(value: str) -> str:
    """Apply lossless text cleanup while retaining accents and punctuation."""
    value = unicodedata.normalize("NFC", value)
    value = value.translate({ord(character): None for character in ZERO_WIDTH_CHARACTERS})
    return " ".join(value.split())


def dedupe_key(value: str) -> str:
    """Collapse only case and whitespace differences, not accents or punctuation."""
    return normalize_address(value).casefold()


def classify_signals(text: str) -> list[str]:
    """Attach coarse sampling signals; these are not model labels."""
    signals: list[str] = []
    if HOUSE_NUMBER_PATTERN.search(text):
        signals.append("HAS_HOUSE_NUMBER")
    else:
        signals.append("MISSING_HOUSE_NUMBER")
    if text.isascii() and any(character.isalpha() for character in text):
        signals.append("ASCII_ONLY")
    if ABBREVIATION_PATTERN.search(text):
        signals.append("ABBREVIATED")
    if WARD_MARKER_PATTERN.search(text):
        signals.append("HAS_WARD_MARKER")
    if DISTRICT_MARKER_PATTERN.search(text):
        signals.append("HAS_LEGACY_DISTRICT")
    if PROVINCE_MARKER_PATTERN.search(text):
        signals.append("HAS_PROVINCE_MARKER")
    if not any(
        pattern.search(text)
        for pattern in (
            WARD_MARKER_PATTERN,
            DISTRICT_MARKER_PATTERN,
            PROVINCE_MARKER_PATTERN,
        )
    ):
        signals.append("NO_ADMIN_MARKER")
    if "," not in text and ";" not in text:
        signals.append("NO_DELIMITERS")
    return signals


def rejection_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    if len(text) < 4:
        reasons.append("too_short")
    if len(text) > 300:
        reasons.append("too_long")
    if not any(character.isalpha() for character in text):
        reasons.append("no_letters")
    if EMAIL_PATTERN.search(text):
        reasons.append("contains_email")
    if PHONE_PATTERN.search(text):
        reasons.append("contains_phone")
    return reasons


def read_raw_addresses(path: Path, column: str | None = None) -> list[tuple[int, str]]:
    """Read one-address-per-line TXT or one address column from CSV."""
    suffix = path.suffix.casefold()
    if suffix in {".txt", ".text"}:
        return list(enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1))
    if suffix != ".csv":
        raise ValueError("input must be a .txt or .csv file")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        selected_column = column or _detect_address_column(fieldnames)
        if selected_column not in fieldnames:
            raise ValueError(
                f"CSV column {selected_column!r} not found; available columns: {fieldnames}"
            )
        return [
            (row_number, row.get(selected_column, "") or "")
            for row_number, row in enumerate(reader, start=2)
        ]


def _detect_address_column(fieldnames: list[str]) -> str:
    normalized_fields = {field.strip().casefold(): field for field in fieldnames}
    for candidate in DEFAULT_ADDRESS_COLUMNS:
        if candidate.casefold() in normalized_fields:
            return normalized_fields[candidate.casefold()]
    raise ValueError(
        "cannot detect address column; pass --column explicitly. "
        f"Available columns: {fieldnames}"
    )


def curate_addresses(
    rows: Iterable[tuple[int, str]], source_name: str
) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: dict[str, str] = {}

    for source_row, raw_value in rows:
        text = normalize_address(str(raw_value))
        reasons = rejection_reasons(text)
        if not text and "too_short" not in reasons:
            reasons.append("too_short")
        if reasons:
            rejected.append(
                {
                    "source": source_name,
                    "source_row": source_row,
                    "text": text,
                    "reasons": reasons,
                }
            )
            continue

        key = dedupe_key(text)
        if key in seen:
            rejected.append(
                {
                    "source": source_name,
                    "source_row": source_row,
                    "text": text,
                    "reasons": ["duplicate"],
                    "duplicate_of": seen[key],
                }
            )
            continue

        record_id = f"curated_{len(accepted) + 1:06d}"
        seen[key] = record_id
        accepted.append(
            {
                "id": record_id,
                "text": text,
                "source": source_name,
                "source_row": source_row,
                "signals": classify_signals(text),
            }
        )
    return accepted, rejected


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_report(accepted: list[dict], rejected: list[dict]) -> dict:
    signal_counts = Counter(
        signal for record in accepted for signal in record["signals"]
    )
    rejection_counts = Counter(
        reason for record in rejected for reason in record["reasons"]
    )
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "signal_counts": dict(sorted(signal_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def main() -> int:
    source_name = SOURCE_NAME or INPUT_PATH.stem
    rows = read_raw_addresses(INPUT_PATH, CSV_ADDRESS_COLUMN)
    accepted, rejected = curate_addresses(rows, source_name)
    report = build_report(accepted, rejected)

    write_jsonl(OUTPUT_PATH, accepted)
    write_jsonl(REJECTED_OUTPUT_PATH, rejected)
    TEXT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEXT_OUTPUT_PATH.write_text(
        "".join(f"{record['text']}\n" for record in accepted), encoding="utf-8"
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"accepted JSONL: {OUTPUT_PATH}")
    print(f"annotation input: {TEXT_OUTPUT_PATH}")
    print(f"rejected rows: {REJECTED_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
