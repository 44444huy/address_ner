import unittest

from address_ner.evaluate_resolver import (
    calibrate_thresholds,
    evaluate_resolution_cases,
)
from address_ner.prepare_annotation import AdminUnit
from address_ner.resolver import AdminResolver


def unit(
    province_code: str,
    province_name: str,
    ward_code: str,
    ward_name: str,
    unit_type: str,
    short_name: str,
) -> AdminUnit:
    return AdminUnit(
        province_code=province_code,
        province_name=province_name,
        ward_code=ward_code,
        ward_name=ward_name,
        unit_type=unit_type,
        short_name=short_name,
    )


class ResolverEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.units = [
            unit("01", "Thành phố Hà Nội", "00004", "Phường Ba Đình", "PHƯỜNG", "Ba Đình"),
            unit("79", "Thành phố Hồ Chí Minh", "26743", "Phường Bến Thành", "PHƯỜNG", "Bến Thành"),
        ]
        self.cases = [
            {
                "id": "correct_typo",
                "categories": ["ward_typo"],
                "entities": [
                    {"type": "WARD", "text": "P. Ben Thnah"},
                    {"type": "PROVINCE", "text": "TP HCM"},
                ],
                "expected": {
                    "accept": True,
                    "ward_code": "26743",
                    "province_code": "79",
                },
            },
            {
                "id": "reject_unknown",
                "categories": ["negative"],
                "entities": [
                    {"type": "WARD", "text": "P. Zzzzzz"},
                    {"type": "PROVINCE", "text": "TP HCM"},
                ],
                "expected": {"accept": False, "status": "REJECTED"},
            },
        ]

    def test_reports_precision_coverage_and_category_slices(self) -> None:
        report = evaluate_resolution_cases(AdminResolver(self.units), self.cases)
        self.assertEqual(report["num_cases"], 2)
        self.assertEqual(report["num_false_accepts"], 0)
        self.assertEqual(report["accept_precision"], 1.0)
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["decision_accuracy"], 1.0)
        self.assertEqual(report["failed_cases"], [])
        self.assertIn("ward_typo", report["per_category"])

    def test_calibration_ranks_zero_false_accepts_first(self) -> None:
        ranking = calibrate_thresholds(
            self.units,
            self.cases,
            ward_thresholds=[0.8, 0.95],
            province_thresholds=[0.82],
            margins=[0.03],
        )
        self.assertEqual(ranking[0]["num_false_accepts"], 0)
        self.assertGreaterEqual(
            ranking[0]["num_correct_accepts"],
            ranking[-1]["num_correct_accepts"],
        )


if __name__ == "__main__":
    unittest.main()
