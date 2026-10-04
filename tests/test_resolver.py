import unittest

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


class AdminResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = AdminResolver(
            [
                unit("01", "Thành phố Hà Nội", "00004", "Phường Ba Đình", "PHƯỜNG", "Ba Đình"),
                unit("01", "Thành phố Hà Nội", "00008", "Phường Trung Sơn", "PHƯỜNG", "Trung Sơn"),
                unit("79", "Thành phố Hồ Chí Minh", "26734", "Phường Trung Sơn", "PHƯỜNG", "Trung Sơn"),
                unit("79", "Thành phố Hồ Chí Minh", "26740", "Phường Bến Thành", "PHƯỜNG", "Bến Thành"),
            ]
        )

    def test_resolves_exact_ward_with_parent_province(self) -> None:
        result = self.resolver.resolve(
            [
                {"type": "WARD", "text": "Phường Ba Đình", "mean_confidence": 0.9},
                {"type": "PROVINCE", "text": "Hà Nội", "mean_confidence": 0.8},
            ]
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["current"]["ward"]["code"], "00004")
        self.assertEqual(result["current"]["province"]["code"], "01")
        self.assertAlmostEqual(result["parser_confidence"], 0.85)

    def test_resolves_unaccented_abbreviations(self) -> None:
        result = self.resolver.resolve(
            [
                {"type": "WARD", "text": "P. Ben Thanh"},
                {"type": "PROVINCE", "text": "TP HCM"},
            ]
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["current"]["ward"]["code"], "26740")
        self.assertEqual(result["resolver_confidence"], 0.9)

    def test_parent_province_disambiguates_duplicate_ward(self) -> None:
        result = self.resolver.resolve(
            [
                {"type": "WARD", "text": "Phường Trung Sơn"},
                {"type": "PROVINCE", "text": "Hà Nội"},
            ]
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["current"]["ward"]["code"], "00008")

    def test_duplicate_ward_without_province_is_ambiguous(self) -> None:
        result = self.resolver.resolve(
            [{"type": "WARD", "text": "Phường Trung Sơn"}]
        )
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertEqual(result["candidate_ward_codes"], ["00008", "26734"])

    def test_rejects_ward_province_conflict(self) -> None:
        result = self.resolver.resolve(
            [
                {"type": "WARD", "text": "Phường Bến Thành"},
                {"type": "PROVINCE", "text": "Hà Nội"},
            ]
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn(
            "WARD_PROVINCE_CONFLICT",
            {issue["code"] for issue in result["issues"]},
        )

    def test_infers_parent_for_unique_ward_with_warning(self) -> None:
        result = self.resolver.resolve(
            [{"type": "WARD", "text": "P. Ba Đình"}]
        )
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["current"]["province"]["code"], "01")
        self.assertIn(
            "PROVINCE_INFERRED_FROM_WARD",
            {issue["code"] for issue in result["issues"]},
        )

    def test_fuzzy_matches_typo_with_warning(self) -> None:
        result = self.resolver.resolve(
            [
                {"type": "WARD", "text": "P. Ben Thnah"},
                {"type": "PROVINCE", "text": "TP HCM"},
            ]
        )
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["current"]["ward"]["code"], "26740")
        self.assertEqual(result["current"]["ward"]["match_method"], "fuzzy_alias")
        self.assertGreaterEqual(result["resolver_confidence"], 0.84)
        self.assertIn(
            "WARD_FUZZY_MATCH",
            {issue["code"] for issue in result["issues"]},
        )

    def test_rejects_unrelated_ward_below_fuzzy_threshold(self) -> None:
        result = self.resolver.resolve(
            [
                {"type": "WARD", "text": "P. Zzzzzz"},
                {"type": "PROVINCE", "text": "TP HCM"},
            ]
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertIsNone(result["current"]["ward"])
        self.assertEqual(result["resolver_confidence"], 0.0)
        self.assertIn(
            "WARD_NOT_FOUND",
            {issue["code"] for issue in result["issues"]},
        )

    def test_fuzzy_matches_province_typo_with_warning(self) -> None:
        result = self.resolver.resolve(
            [
                {"type": "WARD", "text": "P. Ben Thanh"},
                {"type": "PROVINCE", "text": "TP Ho Chi Mnih"},
            ]
        )
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(result["current"]["province"]["code"], "79")
        self.assertEqual(
            result["current"]["province"]["match_method"], "fuzzy_alias"
        )
        self.assertIn(
            "PROVINCE_FUZZY_MATCH",
            {issue["code"] for issue in result["issues"]},
        )

    def test_close_fuzzy_candidates_are_ambiguous(self) -> None:
        resolver = AdminResolver(
            [
                unit("79", "Thành phố Hồ Chí Minh", "10001", "Phường Minh An", "PHƯỜNG", "Minh An"),
                unit("79", "Thành phố Hồ Chí Minh", "10002", "Phường Minh Anh", "PHƯỜNG", "Minh Anh"),
            ],
            ward_fuzzy_threshold=0.7,
            fuzzy_margin=0.2,
        )
        result = resolver.resolve(
            [
                {"type": "WARD", "text": "P. Minh Ah"},
                {"type": "PROVINCE", "text": "TP HCM"},
            ]
        )
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertEqual(result["candidate_ward_codes"], ["10001", "10002"])
        self.assertEqual(result["resolver_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
