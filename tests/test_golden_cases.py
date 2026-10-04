import json
import tempfile
import unittest
from pathlib import Path

from address_ner.golden_cases import (
    validate_golden_jsonl,
    validate_golden_record,
)


class GoldenCaseTests(unittest.TestCase):
    def make_record(self) -> dict:
        return {
            "id": "golden_test",
            "text": "Phường Bến Thành",
            "label": [[0, 16, "WARD"]],
            "annotation_status": "approved",
            "source": "manual",
            "categories": ["PARTIAL"],
        }

    def test_accepts_valid_case(self) -> None:
        self.assertEqual(validate_golden_record(self.make_record()), [])

    def test_rejects_unknown_category(self) -> None:
        record = self.make_record()
        record["categories"] = ["EASY"]
        issues = validate_golden_record(record)
        self.assertTrue(any("unknown golden category" in issue.message for issue in issues))

    def test_rejects_entities_on_invalid_case(self) -> None:
        record = self.make_record()
        record["categories"] = ["INVALID"]
        issues = validate_golden_record(record)
        self.assertTrue(any("must not contain" in issue.message for issue in issues))

    def test_detects_duplicate_text(self) -> None:
        first = self.make_record()
        second = {**first, "id": "golden_test_2"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.jsonl"
            with path.open("w", encoding="utf-8") as file:
                file.write(json.dumps(first, ensure_ascii=False) + "\n")
                file.write(json.dumps(second, ensure_ascii=False) + "\n")
            _, _, issues = validate_golden_jsonl(path)
        self.assertTrue(any("duplicate text" in issue.message for issue in issues))


if __name__ == "__main__":
    unittest.main()
