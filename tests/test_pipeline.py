import unittest

from address_ner.legacy_resolver import LegacyAdminResolver, LegacyComparison
from address_ner.pipeline import (
    build_pipeline_output,
    repair_legacy_district_prediction,
)


class PipelineOutputTests(unittest.TestCase):
    def test_builds_normalized_contract_from_entities_and_resolution(self) -> None:
        entities = [
            {"type": "HOUSE_NUMBER", "text": "12", "mean_confidence": 0.9},
            {"type": "STREET", "text": "Nguyễn Trãi", "mean_confidence": 0.8},
            {"type": "WARD", "text": "Phường Ba Đình", "mean_confidence": 0.95},
            {"type": "PROVINCE", "text": "Hà Nội", "mean_confidence": 0.85},
        ]
        resolution = {
            "status": "PASS",
            "current": {
                "ward": {"code": "00004", "name": "Phường Ba Đình"},
                "province": {"code": "01", "name": "Thành phố Hà Nội"},
            },
            "resolver_confidence": 1.0,
            "candidate_ward_codes": ["00004"],
            "candidate_wards": [
                {
                    "code": "00004",
                    "name": "Phường Ba Đình",
                    "province_code": "01",
                    "score": 1.0,
                }
            ],
            "issues": [],
        }

        result = build_pipeline_output(
            "12 Nguyễn Trãi, Phường Ba Đình, Hà Nội",
            "12 Nguyễn Trãi, Phường Ba Đình, Hà Nội",
            entities,
            resolution,
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["detected_structure"], "current_two_level")
        self.assertEqual(result["components"]["street"], "Nguyễn Trãi")
        self.assertEqual(result["current"]["ward"]["code"], "00004")
        self.assertIsNone(result["legacy"])
        self.assertEqual(result["candidate_wards"][0]["score"], 1.0)
        self.assertAlmostEqual(result["parser_confidence"], 0.875)

    def test_marks_district_structure_as_legacy(self) -> None:
        entities = [
            {"type": "DISTRICT", "text": "Quận 1", "mean_confidence": 0.9}
        ]
        resolution = {
            "status": "REJECTED",
            "current": {"ward": None, "province": None},
            "resolver_confidence": 0.0,
            "candidate_ward_codes": [],
            "issues": [],
        }
        result = build_pipeline_output("Quận 1", "Quận 1", entities, resolution)
        self.assertEqual(result["detected_structure"], "legacy_with_district")

    def test_repairs_legacy_city_district_predicted_as_province(self) -> None:
        resolver = LegacyAdminResolver(
            [
                LegacyComparison(
                    legacy_province_code="08",
                    legacy_province_name="Tỉnh Tuyên Quang",
                    legacy_district_code="070",
                    legacy_district_name="Thành phố Tuyên Quang",
                    legacy_ward_code="02209",
                    legacy_ward_name="Xã Tràng Đà",
                    legacy_effective_from="",
                    current_ward_code="",
                    current_ward_name="",
                    current_province_code="",
                    current_province_name="",
                    current_effective_from="",
                    note="",
                    source_url="",
                )
            ]
        )
        entities = [
            {"type": "WARD", "text": "Xã Tràng Đà"},
            {"type": "PROVINCE", "text": "Thành phố Tuyên Quang"},
            {"type": "PROVINCE", "text": "Tỉnh Tuyên Quang"},
        ]

        repaired = repair_legacy_district_prediction(entities, resolver)

        self.assertEqual(repaired[1]["type"], "DISTRICT")
        self.assertEqual(
            repaired[1]["structure_repaired"],
            "legacy_district_from_province",
        )
        self.assertEqual(repaired[2]["type"], "PROVINCE")
        self.assertNotIn("structure_repaired", entities[1])

    def test_does_not_repair_a_single_current_province(self) -> None:
        entities = [
            {"type": "WARD", "text": "Phường Ba Đình"},
            {"type": "PROVINCE", "text": "Thành phố Hà Nội"},
        ]

        repaired = repair_legacy_district_prediction(entities, None)

        self.assertEqual(repaired, entities)


if __name__ == "__main__":
    unittest.main()
