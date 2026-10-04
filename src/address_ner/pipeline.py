"""Run raw Vietnamese address text through PhoBERT NER and admin resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .build_legacy_mappings import load_legacy_mappings
from .convert_annotations import VnCoreNlpWordSegmenter
from .curate_raw_addresses import normalize_address
from .inference import predict_record
from .legacy_resolver import LegacyAdminResolver, load_legacy_comparisons
from .prepare_annotation import load_catalog
from .resolver import AdminResolver


# Configuration: edit these values to run the end-to-end pipeline.
INPUT_TEXT = "12 Nguyen Trai, P. Ba Dinh, TP Ha Noi"
CHECKPOINT_PATH = Path("checkpoints/synthetic-v3")
CATALOG_PATH = Path("data/catalog/admin_units_2025.csv")
VNCORENLP_MODEL_DIR = Path("tools/vncorenlp")
LEGACY_CATALOG_PATH = Path("data/catalog/ward_comparison_2025.csv")
LEGACY_MAPPING_CATALOG_PATH = Path("data/catalog/legacy_mappings_2025.csv")
MAX_LENGTH = 64
FORCE_CPU = False


def _entity_confidence(entity: dict[str, Any]) -> float:
    return float(entity.get("mean_confidence", entity.get("confidence", 0.0)))


def _best_entity(
    entities: list[dict[str, Any]], entity_type: str
) -> dict[str, Any] | None:
    candidates = [entity for entity in entities if entity["type"] == entity_type]
    return max(candidates, key=_entity_confidence) if candidates else None


_LEGACY_DISTRICT_PREFIX = re.compile(
    r"^(Thành phố|Thị xã|Quận|Huyện)\b",
    flags=re.IGNORECASE,
)


def repair_legacy_district_prediction(
    entities: list[dict[str, Any]],
    legacy_resolver: LegacyAdminResolver | None,
) -> list[dict[str, Any]]:
    if legacy_resolver is None or any(
        entity.get("type") == "DISTRICT" for entity in entities
    ):
        return entities

    province_indices = [
        index
        for index, entity in enumerate(entities)
        if entity.get("type") == "PROVINCE"
    ]
    if len(province_indices) < 2:
        return entities

    successful_repairs: list[int] = []
    for index in province_indices:
        if _LEGACY_DISTRICT_PREFIX.match(entities[index].get("text", "")) is None:
            continue
        trial = [dict(entity) for entity in entities]
        trial[index]["type"] = "DISTRICT"
        resolution = legacy_resolver.resolve(trial)
        if resolution.get("legacy") is not None:
            successful_repairs.append(index)

    if len(successful_repairs) != 1:
        return entities

    repaired = [dict(entity) for entity in entities]
    repaired[successful_repairs[0]]["type"] = "DISTRICT"
    repaired[successful_repairs[0]][
        "structure_repaired"
    ] = "legacy_district_from_province"
    return repaired


def build_pipeline_output(
    raw_text: str,
    normalized_input: str,
    predicted_entities: list[dict[str, Any]],
    resolution: dict[str, Any],
) -> dict[str, Any]:
    selected = {
        entity_type: _best_entity(predicted_entities, entity_type)
        for entity_type in (
            "HOUSE_NUMBER",
            "STREET",
            "WARD",
            "DISTRICT",
            "PROVINCE",
        )
    }
    confidences = [_entity_confidence(entity) for entity in predicted_entities]
    if selected["DISTRICT"] is not None:
        detected_structure = "legacy_with_district"
    elif selected["WARD"] is not None and selected["PROVINCE"] is not None:
        detected_structure = "current_two_level"
    else:
        detected_structure = "partial_or_unknown"

    return {
        "raw_text": raw_text,
        "normalized_input": normalized_input,
        "status": resolution["status"],
        "detected_structure": detected_structure,
        "components": {
            "house_number": (
                selected["HOUSE_NUMBER"]["text"]
                if selected["HOUSE_NUMBER"]
                else None
            ),
            "street": selected["STREET"]["text"] if selected["STREET"] else None,
            "ward": selected["WARD"]["text"] if selected["WARD"] else None,
            "district": (
                selected["DISTRICT"]["text"] if selected["DISTRICT"] else None
            ),
            "province": (
                selected["PROVINCE"]["text"] if selected["PROVINCE"] else None
            ),
        },
        "current": resolution["current"],
        "legacy": resolution.get("legacy"),
        "parser_confidence": (
            sum(confidences) / len(confidences) if confidences else 0.0
        ),
        "resolver_confidence": resolution["resolver_confidence"],
        "candidate_ward_codes": resolution["candidate_ward_codes"],
        "candidate_wards": resolution.get("candidate_wards", []),
        "candidate_legacy_ward_codes": resolution.get(
            "candidate_legacy_ward_codes", []
        ),
        "issues": resolution["issues"],
        "parsed_entities": predicted_entities,
    }


class AddressPipeline:
    def __init__(
        self,
        checkpoint: Path,
        catalog: Path,
        vncorenlp_model_dir: Path,
        legacy_catalog: Path | None = None,
        legacy_mapping_catalog: Path | None = None,
        device: torch.device | None = None,
        max_length: int = 64,
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = max_length
        self.segmenter = VnCoreNlpWordSegmenter(vncorenlp_model_dir.resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, local_files_only=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.checkpoint, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.resolver = AdminResolver(load_catalog(catalog.resolve()))
        if legacy_catalog is not None and legacy_catalog.exists():
            mappings = (
                load_legacy_mappings(legacy_mapping_catalog.resolve())
                if legacy_mapping_catalog is not None
                and legacy_mapping_catalog.exists()
                else None
            )
            self.legacy_resolver = LegacyAdminResolver(
                load_legacy_comparisons(legacy_catalog.resolve()),
                mappings,
            )
        else:
            self.legacy_resolver = None

    def predict(self, raw_text: str) -> dict[str, Any]:
        normalized_input = normalize_address(raw_text)
        if not normalized_input:
            resolution = self.resolver.resolve([])
            return build_pipeline_output(raw_text, normalized_input, [], resolution)

        tokens = self.segmenter.segment(normalized_input)
        record = {
            "id": "inference",
            "raw_text": normalized_input,
            "tokens": tokens,
            "ner_tags": ["O"] * len(tokens),
        }
        prediction = predict_record(
            self.model,
            self.tokenizer,
            record,
            self.device,
            self.max_length,
        )
        entities = repair_legacy_district_prediction(
            prediction["predicted_entities"],
            self.legacy_resolver,
        )
        has_legacy_district = any(
            entity["type"] == "DISTRICT" for entity in entities
        )
        if has_legacy_district and self.legacy_resolver is not None:
            resolution = self.legacy_resolver.resolve(entities)
        else:
            resolution = self.resolver.resolve(entities)
        return build_pipeline_output(
            raw_text,
            normalized_input,
            entities,
            resolution,
        )


def main() -> int:
    device = torch.device("cpu") if FORCE_CPU else None
    pipeline = AddressPipeline(
        checkpoint=CHECKPOINT_PATH,
        catalog=CATALOG_PATH,
        vncorenlp_model_dir=VNCORENLP_MODEL_DIR,
        legacy_catalog=LEGACY_CATALOG_PATH,
        legacy_mapping_catalog=LEGACY_MAPPING_CATALOG_PATH,
        device=device,
        max_length=MAX_LENGTH,
    )
    print(json.dumps(pipeline.predict(INPUT_TEXT), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
