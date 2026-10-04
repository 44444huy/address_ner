"""Load a fine-tuned checkpoint and decode word-level BIO entities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .labels import ID2LABEL
from .resolver import fold_accents
from .tokenize_dataset import load_jsonl, tokenize_and_align_record


# Configuration: edit these values before running inference.
DATASET_PATH = Path("data/synthetic/processed-v2-stratified/test.jsonl")
CHECKPOINT_PATH = Path("checkpoints/synthetic-v3")
MAX_LENGTH = 64


def decode_word_predictions(
    original_words: list[str],
    word_ids: list[int | None],
    logits: torch.Tensor,
) -> list[dict[str, Any]]:
    """Keep the prediction from only the first subword of each original word."""
    probabilities = torch.softmax(logits, dim=-1)
    seen_word_ids: set[int] = set()
    predictions: list[dict[str, Any]] = []

    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids:
            continue

        seen_word_ids.add(word_id)
        label_id = int(probabilities[token_index].argmax().item())
        confidence = float(probabilities[token_index, label_id].item())
        predictions.append(
            {
                "word_index": word_id,
                "token": original_words[word_id],
                "label": ID2LABEL[label_id],
                "confidence": confidence,
            }
        )

    return predictions


def merge_bio_entities(word_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge word-level BIO predictions and repair invalid I-X starts."""
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return

        confidences = current.pop("confidences")
        current["text"] = " ".join(
            token.replace("_", " ") for token in current["tokens"]
        )
        current["mean_confidence"] = sum(confidences) / len(confidences)
        current["min_confidence"] = min(confidences)
        entities.append(current)
        current = None

    for prediction in word_predictions:
        label = prediction["label"]
        if label == "O":
            flush()
            continue

        prefix, entity_type = label.split("-", maxsplit=1)
        can_continue = (
            prefix == "I"
            and current is not None
            and current["type"] == entity_type
        )

        if can_continue:
            current["tokens"].append(prediction["token"])
            current["confidences"].append(prediction["confidence"])
            current["end_word"] = prediction["word_index"]
            continue

        flush()
        current = {
            "type": entity_type,
            "tokens": [prediction["token"]],
            "confidences": [prediction["confidence"]],
            "start_word": prediction["word_index"],
            "end_word": prediction["word_index"],
            "bio_repaired": prefix == "I",
        }

    flush()
    return repair_split_district_entities(entities)


DISTRICT_PREFIXES = {
    "huyen",
    "h",
    "quan",
    "q",
    "thi xa",
    "tx",
    "thanh pho",
    "tp",
}


def repair_split_district_entities(
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join a district prefix with an adjacent BIO-repaired name entity."""
    repaired: list[dict[str, Any]] = []
    index = 0
    while index < len(entities):
        current = entities[index]
        next_entity = entities[index + 1] if index + 1 < len(entities) else None
        prefix = fold_accents(current["text"])
        can_join = (
            current["type"] == "DISTRICT"
            and prefix in DISTRICT_PREFIXES
            and next_entity is not None
            and next_entity["type"] in {"WARD", "DISTRICT"}
            and next_entity.get("bio_repaired", False)
            and next_entity["start_word"] == current["end_word"] + 1
        )
        if not can_join:
            repaired.append(current)
            index += 1
            continue

        current_token_count = len(current["tokens"])
        next_token_count = len(next_entity["tokens"])
        total_tokens = current_token_count + next_token_count
        joined = {
            **current,
            "tokens": current["tokens"] + next_entity["tokens"],
            "end_word": next_entity["end_word"],
            "bio_repaired": True,
            "structure_repaired": "split_district_name",
            "text": f"{current['text']} {next_entity['text']}",
            "mean_confidence": (
                current["mean_confidence"] * current_token_count
                + next_entity["mean_confidence"] * next_token_count
            )
            / total_tokens,
            "min_confidence": min(
                current["min_confidence"],
                next_entity["min_confidence"],
            ),
        }
        repaired.append(joined)
        index += 2
    return repaired


def gold_word_predictions(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "word_index": index,
            "token": token,
            "label": label,
            "confidence": 1.0,
        }
        for index, (token, label) in enumerate(zip(record["tokens"], record["ner_tags"]))
    ]


def predict_record(
    model: torch.nn.Module,
    tokenizer,
    record: dict[str, Any],
    device: torch.device,
    max_length: int,
) -> dict[str, Any]:
    features = tokenize_and_align_record(record, tokenizer, max_length=max_length)
    input_ids = torch.tensor([features["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [features["attention_mask"]], dtype=torch.long, device=device
    )

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0].cpu()

    word_predictions = decode_word_predictions(
        original_words=record["tokens"],
        word_ids=features["word_ids"],
        logits=logits,
    )
    return {
        "id": record["id"],
        "raw_text": record["raw_text"],
        "predicted_words": word_predictions,
        "predicted_entities": merge_bio_entities(word_predictions),
        "gold_entities": merge_bio_entities(gold_word_predictions(record)),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT_PATH,
        local_files_only=True,
    ).to(device)
    model.eval()

    print(f"device={device}, checkpoint={CHECKPOINT_PATH}")
    for record in load_jsonl(DATASET_PATH):
        result = predict_record(model, tokenizer, record, device, MAX_LENGTH)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
