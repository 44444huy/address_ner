import unittest

from address_ner.evaluate_golden import build_golden_report


class GoldenEvaluationTests(unittest.TestCase):
    def test_groups_metrics_by_every_category(self) -> None:
        result = {
            "id": "golden_1",
            "raw_text": "P. Ben Thanh",
            "gold_entities": [
                {"type": "WARD", "start_word": 0, "end_word": 2, "text": "P. Ben Thanh"}
            ],
            "predicted_entities": [
                {
                    "type": "WARD",
                    "start_word": 0,
                    "end_word": 2,
                    "text": "P. Ben Thanh",
                    "mean_confidence": 0.9,
                }
            ],
        }
        report = build_golden_report(
            [result], {"golden_1": ["WITHOUT_ACCENTS", "ABBREVIATED"]}
        )

        self.assertEqual(report["exact_case_accuracy"], 1.0)
        self.assertEqual(report["per_category"]["WITHOUT_ACCENTS"]["num_cases"], 1)
        self.assertEqual(report["per_category"]["ABBREVIATED"]["num_cases"], 1)
        self.assertEqual(
            report["per_category"]["ABBREVIATED"]["exact_case_accuracy"], 1.0
        )
        self.assertEqual(report["failed_cases"], [])

    def test_reports_non_exact_case(self) -> None:
        result = {
            "id": "golden_2",
            "raw_text": "Hà Nội",
            "gold_entities": [
                {"type": "PROVINCE", "start_word": 0, "end_word": 0, "text": "Hà Nội"}
            ],
            "predicted_entities": [],
        }
        report = build_golden_report([result], {"golden_2": ["PARTIAL"]})

        self.assertEqual(report["exact_case_accuracy"], 0.0)
        self.assertEqual(report["entity_metrics"]["overall"]["fn"], 1)
        self.assertEqual(report["failed_cases"][0]["id"], "golden_2")


if __name__ == "__main__":
    unittest.main()
