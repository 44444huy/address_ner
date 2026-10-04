import unittest

from address_ner.build_legacy_mappings import LegacyMapping
from address_ner.evaluate_legacy_mappings import evaluate_mapping_cases


def mapping(
    legacy_code: str,
    current_code: str,
    method: str,
) -> LegacyMapping:
    return LegacyMapping(
        legacy_province_code="01",
        legacy_district_code="001",
        legacy_ward_code=legacy_code,
        legacy_ward_name="Xã Cũ",
        current_province_code="01",
        current_province_name="Tỉnh Mới",
        current_ward_code=current_code,
        current_ward_name="Xã Mới",
        mapping_method=method,
        evidence="official",
        source_url="official",
    )


class LegacyMappingEvaluationTests(unittest.TestCase):
    def test_reports_acceptance_and_category_metrics(self) -> None:
        mappings = [
            mapping("00001", "90001", "NOTE_FULL"),
            mapping("00002", "90002", "NOTE_PARTIAL"),
        ]
        cases = [
            {
                "id": "full",
                "legacy_ward_code": "00001",
                "categories": ["note_full"],
                "expected": {
                    "status": "WARNING",
                    "current_ward_code": "90001",
                    "candidate_ward_codes": ["90001"],
                },
            },
            {
                "id": "partial",
                "legacy_ward_code": "00002",
                "categories": ["partial"],
                "expected": {
                    "status": "AMBIGUOUS",
                    "current_ward_code": None,
                    "candidate_ward_codes": ["90002"],
                },
            },
        ]

        report = evaluate_mapping_cases(mappings, cases)

        self.assertEqual(report["num_cases"], 2)
        self.assertEqual(report["num_false_accepts"], 0)
        self.assertEqual(report["accept_precision"], 1.0)
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["decision_accuracy"], 1.0)
        self.assertEqual(report["failed_cases"], [])
        self.assertIn("partial", report["per_category"])

    def test_counts_wrong_automatic_mapping_as_false_accept(self) -> None:
        report = evaluate_mapping_cases(
            [mapping("00001", "90001", "DIRECT_CODE")],
            [
                {
                    "id": "should_reject",
                    "legacy_ward_code": "00001",
                    "categories": ["negative"],
                    "expected": {
                        "status": "REJECTED",
                        "current_ward_code": None,
                        "candidate_ward_codes": [],
                    },
                }
            ],
        )

        self.assertEqual(report["num_false_accepts"], 1)
        self.assertEqual(report["decision_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
