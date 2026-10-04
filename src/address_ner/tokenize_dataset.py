"""Convert validated word-level records into PhoBERT model features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from transformers import AutoTokenizer

from .labels import LABEL2ID


IGNORE_INDEX = -100
DEFAULT_MODEL_NAME = "vinai/phobert-base-v2"

# Configuration: edit these values before running this module.
DATASET_PATH = Path("data/sample.jsonl")
MODEL_NAME = DEFAULT_MODEL_NAME
MAX_LENGTH = 64
RECORD_LIMIT = 1


class WordTokenizer(Protocol):
    bos_token: str
    eos_token: str
    bos_token_id: int
    eos_token_id: int

    def tokenize(self, text: str) -> list[str]: ...

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...


def tokenize_and_align_record(
    record: dict[str, Any],
    tokenizer: WordTokenizer,
    max_length: int = 64,
) -> dict[str, Any]:
    """Tokenize each word and supervise only its first subword."""
    subword_tokens: list[str] = [tokenizer.bos_token]
    input_ids: list[int] = [tokenizer.bos_token_id]
    attention_mask: list[int] = [1]
    aligned_labels: list[int] = [IGNORE_INDEX]
    word_ids: list[int | None] = [None]

    for word_index, (word, tag) in enumerate(zip(record["tokens"], record["ner_tags"])):
        pieces = tokenizer.tokenize(word)
        if not pieces:
            raise ValueError(f"Tokenizer produced no subword for word {word_index}: {word!r}")

        piece_ids = tokenizer.convert_tokens_to_ids(pieces)
        subword_tokens.extend(pieces)
        input_ids.extend(piece_ids)
        attention_mask.extend([1] * len(pieces))
        word_ids.extend([word_index] * len(pieces))

        aligned_labels.append(LABEL2ID[tag])
        aligned_labels.extend([IGNORE_INDEX] * (len(pieces) - 1))

    subword_tokens.append(tokenizer.eos_token)
    input_ids.append(tokenizer.eos_token_id)
    attention_mask.append(1)
    aligned_labels.append(IGNORE_INDEX)
    word_ids.append(None)

    if len(input_ids) > max_length:
        raise ValueError(
            f"Record {record['id']} has {len(input_ids)} subword tokens, "
            f"exceeding max_length={max_length}"
        )

    lengths = {
        len(subword_tokens),
        len(input_ids),
        len(attention_mask),
        len(aligned_labels),
        len(word_ids),
    }
    if len(lengths) != 1:
        raise RuntimeError(f"Aligned feature lengths differ for record {record['id']}")

    return {
        "id": record["id"],
        "tokens": subword_tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": aligned_labels,
        "word_ids": word_ids,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    records = load_jsonl(DATASET_PATH)

    for record in records[:RECORD_LIMIT]:
        features = tokenize_and_align_record(record, tokenizer, MAX_LENGTH)
        print(json.dumps(features, ensure_ascii=False, indent=2))

    print(f"Tokenized {min(len(records), RECORD_LIMIT)} of {len(records)} record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
