import unittest

from address_ner.build_legacy_mappings import LegacyMapping
from address_ner.legacy_resolver import LegacyAdminResolver, LegacyComparison


def comparison(
    legacy_province_code: str,
    legacy_province_name: str,
    legacy_district_code: str,
    legacy_district_name: str,
    legacy_ward_code: str,
    legacy_ward_name: str,
    current_ward_code: str = "",
    current_ward_name: str = "",
    current_province_code: str = "",
    current_province_name: str = "",
) -> LegacyComparison:
    return LegacyComparison(
        legacy_province_code=legacy_province_code,
        legacy_province_name=legacy_province_name,
        legacy_district_code=legacy_district_code,
        legacy_district_name=legacy_district_name,
        legacy_ward_code=legacy_ward_code,
        legacy_ward_name=legacy_ward_name,
        legacy_effective_from="",
        current_ward_code=current_ward_code,
        current_ward_name=current_ward_name,
        current_province_code=current_province_code,
        current_province_name=current_province_name,
        current_effective_from="",
        note="",
        source_url="",
    )


def mapping(
    legacy_ward_code: str,
    current_ward_code: str,
    current_ward_name: str,
    mapping_method: str,
) -> LegacyMapping:
    return LegacyMapping(
        legacy_province_code="01",
        legacy_district_code="001",
        legacy_ward_code=legacy_ward_code,
        legacy_ward_name="Phường Cũ",
        current_province_code="01",
        current_province_name="Thành phố Hà Nội",
        current_ward_code=current_ward_code,
        current_ward_name=current_ward_name,
        mapping_method=mapping_method,
        evidence="official note",
        source_url="official",
    )


class LegacyAdminResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = LegacyAdminResolver(
            [
                comparison(
                    "04",
                    "Tỉnh Cao Bằng",
                    "042",
                    "Huyện Bảo Lâm",
                    "01304",
                    "Xã Thạch Lâm",
                    "01304",
                    "Xã Quảng Lâm",
                    "04",
                    "Tỉnh Cao Bằng",
                ),
                comparison(
                    "38",
                    "Tỉnh Thanh Hóa",
                    "395",
                    "Huyện Thạch Thành",
                    "15196",
                    "Xã Thạch Lâm",
                ),
                comparison(
                    "01",
                    "Thành phố Hà Nội",
                    "001",
                    "Quận Ba Đình",
                    "00001",
                    "Phường Phúc Xá",
                ),
            ]
        )

    def test_maps_direct_legacy_code_with_hierarchy(self) -> None:
        result = self.resolver.resolve(
            [
                {"type": "WARD", "text": "Xa Thach Lam"},
                {"type": "DISTRICT", "text": "H. Bao Lam"},
                {"type": "PROVINCE", "text": "T. Cao Bang"},
            ]
        )
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["legacy"]["ward"]["code"], "01304")
        self.assertEqual(result["current"]["ward"]["code"], "01304")
        self.assertEqual(result["current"]["ward"]["name"], "Xã Quảng Lâm")

    def test_rejects_identified_legacy_unit_without_direct_mapping(self) -> None:
        result = self.resolver.resolve(
            [
                {"type": "WARD", "text": "Phường Phúc Xá"},
                {"type": "DISTRICT", "text": "Quận Ba Đình"},
                {"type": "PROVINCE", "text": "Hà Nội"},
            ]
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["legacy"]["ward"]["code"], "00001")
        self.assertIsNone(result["current"]["ward"])
        self.assertIn(
            "LEGACY_MAPPING_NOT_AVAILABLE",
            {issue["code"] for issue in result["issues"]},
        )

    def test_is_ambiguous_without_parent_hierarchy(self) -> None:
        result = self.resolver.resolve(
            [{"type": "WARD", "text": "Xã Thạch Lâm"}]
        )
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertEqual(
            result["candidate_legacy_ward_codes"],
            ["01304", "15196"],
        )

    def test_maps_full_note_successor(self) -> None:
        resolver = LegacyAdminResolver(
            [
                comparison(
                    "01",
                    "Thành phố Hà Nội",
                    "001",
                    "Quận Ba Đình",
                    "00001",
                    "Phường Phúc Xá",
                )
            ],
            [mapping("00001", "00097", "Phường Hồng Hà", "NOTE_FULL")],
        )

        result = resolver.resolve(
            [
                {"type": "WARD", "text": "Phường Phúc Xá"},
                {"type": "DISTRICT", "text": "Quận Ba Đình"},
                {"type": "PROVINCE", "text": "Hà Nội"},
            ]
        )

        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["current"]["ward"]["code"], "00097")
        self.assertEqual(
            result["current"]["ward"]["match_method"],
            "legacy_note_full",
        )

    def test_keeps_partial_mapping_ambiguous(self) -> None:
        resolver = LegacyAdminResolver(
            [
                comparison(
                    "01",
                    "Thành phố Hà Nội",
                    "001",
                    "Quận Ba Đình",
                    "00001",
                    "Phường Cũ",
                )
            ],
            [mapping("00001", "00097", "Phường Mới", "NOTE_PARTIAL")],
        )

        result = resolver.resolve(
            [
                {"type": "WARD", "text": "Phường Cũ"},
                {"type": "DISTRICT", "text": "Quận Ba Đình"},
                {"type": "PROVINCE", "text": "Hà Nội"},
            ]
        )

        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertIsNone(result["current"]["ward"])
        self.assertEqual(result["candidate_ward_codes"], ["00097"])

    def test_direct_successor_with_partial_transfer_is_ambiguous(self) -> None:
        resolver = LegacyAdminResolver(
            [
                comparison(
                    "01",
                    "Thành phố Hà Nội",
                    "001",
                    "Quận Ba Đình",
                    "00001",
                    "Phường Cũ",
                    "00001",
                    "Phường Kế Thừa",
                    "01",
                    "Thành phố Hà Nội",
                )
            ],
            [
                mapping("00001", "00001", "Phường Kế Thừa", "DIRECT_CODE"),
                mapping("00001", "00097", "Phường Nhận Một Phần", "NOTE_PARTIAL"),
            ],
        )

        result = resolver.resolve(
            [
                {"type": "WARD", "text": "Phường Cũ"},
                {"type": "DISTRICT", "text": "Quận Ba Đình"},
                {"type": "PROVINCE", "text": "Hà Nội"},
            ]
        )

        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertIsNone(result["current"]["ward"])
        self.assertEqual(result["candidate_ward_codes"], ["00001", "00097"])

    def test_source_note_hint_alone_is_ambiguous(self) -> None:
        resolver = LegacyAdminResolver(
            [
                comparison(
                    "01",
                    "Thành phố Hà Nội",
                    "001",
                    "Quận Ba Đình",
                    "00001",
                    "Phường Cũ",
                )
            ],
            [
                mapping(
                    "00001",
                    "00097",
                    "Phường Được Gợi Ý",
                    "SOURCE_NOTE_HINT",
                )
            ],
        )

        result = resolver.resolve(
            [
                {"type": "WARD", "text": "Phường Cũ"},
                {"type": "DISTRICT", "text": "Quận Ba Đình"},
                {"type": "PROVINCE", "text": "Hà Nội"},
            ]
        )

        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertIsNone(result["current"]["ward"])
        self.assertEqual(result["candidate_ward_codes"], ["00097"])
        self.assertEqual(
            result["issues"][0]["code"], "LEGACY_MAPPING_HINT_ONLY"
        )


if __name__ == "__main__":
    unittest.main()
