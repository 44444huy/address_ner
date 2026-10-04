import random
import unittest

from address_ner.generate_synthetic import (
    build_synthetic_record,
    generate_records,
    remove_accents,
    select_balanced_units,
)
from address_ner.prepare_annotation import AdminUnit
from address_ner.validate_annotations import validate_annotation_record


def make_unit(
    province_code: str = "01",
    province_name: str = "Thành phố Hà Nội",
    ward_code: str = "00004",
    ward_name: str = "Phường Ba Đình",
    unit_type: str = "PHƯỜNG",
    short_name: str = "Ba Đình",
) -> AdminUnit:
    return AdminUnit(
        province_code=province_code,
        province_name=province_name,
        ward_code=ward_code,
        ward_name=ward_name,
        unit_type=unit_type,
        short_name=short_name,
    )


class GenerateSyntheticTests(unittest.TestCase):
    def test_removes_vietnamese_accents_and_d_stroke(self) -> None:
        self.assertEqual(remove_accents("Đường Nguyễn Trãi"), "Duong Nguyen Trai")

    def test_generated_spans_point_to_exact_entities(self) -> None:
        record = build_synthetic_record(
            make_unit(), "Nguyễn Trãi", "synthetic_1", random.Random(4)
        )
        self.assertEqual(validate_annotation_record(record), [])
        for start, end, _ in record["label"]:
            self.assertEqual(record["text"][start:end], record["text"][start:end].strip())

    def test_variants_share_group_and_texts_are_unique(self) -> None:
        records = generate_records(
            [make_unit()], ["Nguyễn Trãi", "Lê Lợi"], count=2, variants_per_unit=2, seed=2
        )
        self.assertEqual(len({record["group_id"] for record in records}), 1)
        self.assertEqual(len({record["text"].casefold() for record in records}), 2)

    def test_balances_small_provinces(self) -> None:
        units = [
            make_unit(ward_code="00001"),
            make_unit(ward_code="00002"),
            make_unit(
                province_code="02",
                province_name="Tỉnh Cao Bằng",
                ward_code="01001",
                ward_name="Phường Thục Phán",
                short_name="Thục Phán",
            ),
        ]
        selected = select_balanced_units(units, 2, random.Random(1))
        self.assertEqual({unit.province_code for unit in selected}, {"01", "02"})


if __name__ == "__main__":
    unittest.main()
