import random
import unittest

from address_ner.generate_hard_cases import (
    build_format_record,
    ensure_no_exact_leakage,
    generate_legacy_records,
    generate_negative_records,
)
from address_ner.prepare_annotation import AdminUnit
from address_ner.validate_annotations import validate_annotation_record


class GenerateHardCasesTests(unittest.TestCase):
    def make_unit(self) -> AdminUnit:
        return AdminUnit(
            province_code="01",
            province_name="Thành phố Hà Nội",
            ward_code="00004",
            ward_name="Phường Ba Đình",
            unit_type="PHƯỜNG",
            short_name="Ba Đình",
        )

    def test_negative_records_have_no_entities(self) -> None:
        records = generate_negative_records(10, random.Random(1))
        self.assertTrue(all(not record["label"] for record in records))
        self.assertTrue(all(validate_annotation_record(record) == [] for record in records))

    def test_legacy_records_supervise_district(self) -> None:
        records = generate_legacy_records(10, ["Nguyễn Trãi"], random.Random(2))
        self.assertTrue(
            all(any(span[2] == "DISTRICT" for span in record["label"]) for record in records)
        )

    def test_format_record_excludes_road_prefix_from_street(self) -> None:
        record = build_format_record(
            self.make_unit(), "Nguyễn Huệ", "hard_1", random.Random(3)
        )
        street_span = next(span for span in record["label"] if span[2] == "STREET")
        self.assertEqual(record["text"][street_span[0] : street_span[1]], "Nguyễn Huệ")
        self.assertIn("đường", record["text"].casefold())

    def test_rejects_exact_golden_leakage(self) -> None:
        record = {"id": "train_1", "text": "Hà Nội"}
        with self.assertRaisesRegex(ValueError, "leakage"):
            ensure_no_exact_leakage([record], [{"id": "gold_1", "text": "HÀ NỘI"}])


if __name__ == "__main__":
    unittest.main()
