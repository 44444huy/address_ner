import unittest

from address_ner.validate_dataset import validate_record


class ValidateRecordTests(unittest.TestCase):
    def make_record(self) -> dict:
        return {
            "id": "addr_test",
            "group_id": "base_test",
            "raw_text": "Phường Bến Thành",
            "tokens": ["Phường", "Bến_Thành"],
            "ner_tags": ["B-WARD", "I-WARD"],
            "source": "manual",
        }

    def test_accepts_valid_record(self) -> None:
        self.assertEqual(validate_record(self.make_record(), 1), [])

    def test_rejects_mismatched_lengths(self) -> None:
        record = self.make_record()
        record["ner_tags"] = ["B-WARD"]
        issues = validate_record(record, 1)
        self.assertIn("does not match", issues[0].message)

    def test_rejects_invalid_i_transition(self) -> None:
        record = self.make_record()
        record["ner_tags"] = ["O", "I-WARD"]
        issues = validate_record(record, 1)
        self.assertIn("does not continue", issues[0].message)


if __name__ == "__main__":
    unittest.main()

