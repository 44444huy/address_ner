import unittest

from address_ner.build_legacy_mappings import (
    ProvinceComparison,
    build_legacy_mappings,
    build_province_successors,
)
from address_ner.legacy_resolver import LegacyComparison
from address_ner.prepare_annotation import AdminUnit


def comparison(
    legacy_code: str,
    legacy_name: str,
    *,
    district_code: str = "001",
    district_name: str = "Huyện Cũ",
    current_code: str = "",
    current_name: str = "",
    note: str = "",
) -> LegacyComparison:
    return LegacyComparison(
        legacy_province_code="01",
        legacy_province_name="Tỉnh Cũ",
        legacy_district_code=district_code,
        legacy_district_name=district_name,
        legacy_ward_code=legacy_code,
        legacy_ward_name=legacy_name,
        legacy_effective_from="",
        current_ward_code=current_code,
        current_ward_name=current_name,
        current_province_code="91" if current_code else "",
        current_province_name="Tỉnh Mới" if current_code else "",
        current_effective_from="",
        note=note,
        source_url="official",
    )


class BuildLegacyMappingsTests(unittest.TestCase):
    def test_builds_province_successor_from_note_destination(self) -> None:
        comparisons = [
            ProvinceComparison(
                "01",
                "Tỉnh Cũ",
                "",
                "",
                "",
                "",
                "Sắp xếp tỉnh Cũ thành tỉnh mới có tên gọi là tỉnh Mới",
                "official",
            )
        ]
        current = [
            AdminUnit("91", "Tỉnh Mới", "00001", "Xã Mới", "XÃ", "Mới")
        ]

        successors = build_province_successors(comparisons, current)

        self.assertEqual(successors["01"].current_province_code, "91")
        self.assertEqual(successors["01"].mapping_method, "NOTE_PROVINCE_NAME")

    def test_classifies_full_partial_and_ignores_destination_mentions(self) -> None:
        old_a = comparison("00001", "Xã Nậm Tha")
        old_b = comparison("00002", "Xã Chiềng Ken")
        old_c = comparison("00003", "Phường Phú Mỹ")
        old_destination = comparison("00004", "Xã Mới")
        target = comparison(
            "",
            "",
            current_code="90001",
            current_name="Xã Mới",
            note=(
                "Hợp nhất xã Nậm Tha và xã Chiềng Ken, 1 phần phường "
                "Phú Mỹ thành xã Mới"
            ),
        )

        mappings = build_legacy_mappings(
            [old_a, old_b, old_c, old_destination, target],
            {"01": type("S", (), {"current_province_code": "91"})()},
        )
        methods = {
            mapping.legacy_ward_code: mapping.mapping_method
            for mapping in mappings
        }

        self.assertEqual(methods["00001"], "NOTE_FULL")
        self.assertEqual(methods["00002"], "NOTE_FULL")
        self.assertEqual(methods["00003"], "NOTE_PARTIAL")
        self.assertNotIn("00004", methods)
        self.assertNotIn("", methods)

    def test_ignores_boundary_description_without_legal_action(self) -> None:
        old = comparison("00001", "Phường Ngọc Hà")
        target = comparison(
            "",
            "",
            current_code="90001",
            current_name="Phường Mới",
            note="Đông giáp phường Ngọc Hà; Tây giáp phường Cống Vị",
        )

        mappings = build_legacy_mappings(
            [old, target],
            {"01": type("S", (), {"current_province_code": "91"})()},
        )

        self.assertEqual(mappings, [])

    def test_keeps_direct_comparison_mapping(self) -> None:
        direct = comparison(
            "00001",
            "Xã Cũ",
            current_code="90001",
            current_name="Xã Mới",
        )

        mappings = build_legacy_mappings(
            [direct],
            {"01": type("S", (), {"current_province_code": "91"})()},
        )

        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0].mapping_method, "DIRECT_CODE")

    def test_uses_target_district_to_disambiguate_duplicate_names(self) -> None:
        correct = comparison(
            "00001",
            "Xã Sơn Thủy",
            district_code="001",
            district_name="Huyện Thanh Thủy",
        )
        duplicate = comparison(
            "00002",
            "Xã Sơn Thủy",
            district_code="002",
            district_name="Huyện Mai Châu",
        )
        target = comparison(
            "00999",
            "Xã Đích",
            district_code="001",
            district_name="Huyện Thanh Thủy",
            current_code="90001",
            current_name="Xã Mới",
            note="Hợp nhất xã Sơn Thủy và xã Đích",
        )

        mappings = build_legacy_mappings(
            [correct, duplicate, target],
            {"01": type("S", (), {"current_province_code": "91"})()},
        )

        note_codes = {
            mapping.legacy_ward_code
            for mapping in mappings
            if mapping.mapping_method == "NOTE_FULL"
        }
        self.assertIn("00001", note_codes)
        self.assertNotIn("00002", note_codes)

    def test_source_note_creates_hint_but_not_full_mapping(self) -> None:
        source = comparison(
            "00001",
            "Xã Bình An",
            note="Đóng mã do sáp nhập vào xã Hồng Thái mới",
        )
        target = comparison(
            "00999",
            "Xã Cũ",
            current_code="90001",
            current_name="Xã Hồng Thái",
        )

        mappings = build_legacy_mappings(
            [source, target],
            {"01": type("S", (), {"current_province_code": "91"})()},
        )

        source_mapping = next(
            mapping
            for mapping in mappings
            if mapping.legacy_ward_code == "00001"
        )
        self.assertEqual(source_mapping.current_ward_code, "90001")
        self.assertEqual(source_mapping.mapping_method, "SOURCE_NOTE_HINT")

    def test_source_note_ignores_unknown_destination(self) -> None:
        source = comparison(
            "00001",
            "Xã Bình An",
            note="Đóng mã do sáp nhập vào xã Không Có",
        )

        mappings = build_legacy_mappings(
            [source],
            {"01": type("S", (), {"current_province_code": "91"})()},
        )

        self.assertEqual(mappings, [])


if __name__ == "__main__":
    unittest.main()
