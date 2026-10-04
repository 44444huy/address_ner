"""Convert approved character-span annotations to word-level BIO JSONL."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .validate_annotations import (
    load_annotation_jsonl,
    validate_annotation_record,
)


# Configuration: edit these values before running this module.
INPUT_PATH = Path("data/synthetic/annotations-v2.jsonl")
OUTPUT_PATH = Path("data/synthetic/bio-v2.jsonl")
MODEL_DIR = Path("tools/vncorenlp")


@dataclass(frozen=True)
class TokenSpan:
    token: str
    start: int
    end: int


class WordSegmenter(Protocol):
    def segment(self, text: str) -> list[str]: ...


ADDRESS_COMPONENT_PATTERN = re.compile(
    r"\b(?:thị\s+xã|thành\s+phố|đặc\s+khu|phường|xã|quận|huyện|tỉnh)\b",
    flags=re.IGNORECASE,
)
ROAD_PREFIX_PATTERN = re.compile(r"\bđường\b", flags=re.IGNORECASE)


def split_address_chunks(text: str) -> list[str]:
    """Split before explicit administrative markers without changing the text."""
    boundaries = {
        match.start()
        for match in ADDRESS_COMPONENT_PATTERN.finditer(text)
        if match.start() > 0
    }
    for match in ROAD_PREFIX_PATTERN.finditer(text):
        boundaries.update({match.start(), match.end()})
    ordered_boundaries = sorted(boundary for boundary in boundaries if 0 < boundary < len(text))
    starts = [0, *ordered_boundaries]
    ends = [*ordered_boundaries, len(text)]
    return [text[start:end].strip() for start, end in zip(starts, ends) if text[start:end].strip()]


class VnCoreNlpWordSegmenter:
    def __init__(self, model_dir: Path) -> None:
        try:
            import py_vncorenlp
        except ImportError as error:
            raise RuntimeError(
                "py_vncorenlp is not installed; install it before BIO conversion"
            ) from error

        original_working_directory = Path.cwd()
        try:
            self.model = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"],
                save_dir=str(model_dir.resolve()),
            )
        finally:
            os.chdir(original_working_directory)

    def segment(self, text: str) -> list[str]:
        tokens: list[str] = []
        for chunk in split_address_chunks(text):
            sentences = self.model.word_segment(chunk)
            tokens.extend(
                token for sentence in sentences for token in sentence.split()
            )
        return tokens


def locate_segmented_tokens(text: str, tokens: list[str]) -> list[TokenSpan]:
    """Map VnCoreNLP tokens with underscores back to raw character offsets."""
    located: list[TokenSpan] = []
    cursor = 0
    for token in tokens:
        surface_parts = token.split("_")
        pattern = r"\s+".join(
            re.escape(_fold_for_alignment(part)) for part in surface_parts
        )
        match = re.search(pattern, _fold_for_alignment(text[cursor:]))
        if match is None:
            raise ValueError(
                f"cannot map segmented token {token!r} after character {cursor} in {text!r}"
            )
        gap = text[cursor : cursor + match.start()]
        if gap.strip():
            raise ValueError(
                f"segmenter skipped non-whitespace text {gap!r} before token {token!r}"
            )
        start = cursor + match.start()
        end = cursor + match.end()
        located.append(TokenSpan(token=token, start=start, end=end))
        cursor = end

    if text[cursor:].strip():
        raise ValueError(f"segmenter did not return trailing text {text[cursor:]!r}")
    return located


def _fold_for_alignment(value: str) -> str:
    """Fold Vietnamese tone variants while preserving one output char per input char."""
    folded: list[str] = []
    for character in unicodedata.normalize("NFC", value):
        if character in {"Đ", "đ"}:
            folded.append("d")
            continue
        base = "".join(
            part
            for part in unicodedata.normalize("NFD", character)
            if unicodedata.category(part) != "Mn"
        )
        folded.append(base.lower())
    return "".join(folded)


def align_char_spans_to_bio(
    text: str,
    token_spans: list[TokenSpan],
    entity_spans: list[list[int | str]],
) -> list[str]:
    labels = ["O"] * len(token_spans)

    for entity_start, entity_end, entity_type in entity_spans:
        start = int(entity_start)
        end = int(entity_end)
        covered_indices: list[int] = []

        for token_index, token_span in enumerate(token_spans):
            overlaps = token_span.start < end and start < token_span.end
            if not overlaps:
                continue
            fully_inside = start <= token_span.start and token_span.end <= end
            if not fully_inside:
                raise ValueError(
                    f"entity {text[start:end]!r} [{start}, {end}) cuts through token "
                    f"{token_span.token!r} [{token_span.start}, {token_span.end})"
                )
            if labels[token_index] != "O":
                raise ValueError(f"token {token_span.token!r} belongs to multiple entities")
            covered_indices.append(token_index)

        if not covered_indices:
            raise ValueError(f"entity {text[start:end]!r} is not covered by any token")

        first_token = token_spans[covered_indices[0]]
        last_token = token_spans[covered_indices[-1]]
        if first_token.start != start or last_token.end != end:
            raise ValueError(
                f"entity {text[start:end]!r} boundaries do not match token boundaries"
            )

        for offset, token_index in enumerate(covered_indices):
            prefix = "B" if offset == 0 else "I"
            labels[token_index] = f"{prefix}-{entity_type}"

    return labels


def convert_record(record: dict[str, Any], segmenter: WordSegmenter) -> dict[str, Any]:
    issues = validate_annotation_record(record)
    if issues:
        raise ValueError("; ".join(issue.message for issue in issues))
    if record["annotation_status"] != "approved":
        raise ValueError(
            f"record {record['id']} is {record['annotation_status']!r}, not 'approved'"
        )

    tokens = segmenter.segment(record["text"])
    token_spans = locate_segmented_tokens(record["text"], tokens)
    ner_tags = align_char_spans_to_bio(record["text"], token_spans, record["label"])
    converted = {
        "id": record["id"],
        "group_id": record.get("group_id", record["id"]),
        "raw_text": record["text"],
        "tokens": tokens,
        "ner_tags": ner_tags,
        "source": record["source"],
    }
    for metadata_field in ("categories", "generation"):
        if metadata_field in record:
            converted[metadata_field] = record[metadata_field]
    return converted


def main() -> int:
    input_path = INPUT_PATH.resolve()
    output_path = OUTPUT_PATH.resolve()
    model_dir = MODEL_DIR.resolve()
    records = load_annotation_jsonl(input_path)
    segmenter = VnCoreNlpWordSegmenter(model_dir)
    converted = [convert_record(record, segmenter) for record in records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in converted:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"converted {len(converted)} approved record(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
