import unittest

from address_ner.fetch_legacy_catalog import (
    build_custom_callback,
    build_report,
    normalize_province_comparison_rows,
    normalize_ward_comparison_rows,
)


class LegacyCatalogImportTests(unittest.TestCase):
    def test_builds_custom_callback(self) -> None:
        self.assertEqual(
            build_custom_callback(),
            "GB|19;14|CUSTOMCALLBACK0|;",
        )

    def test_normalizes_ward_comparison_columns(self) -> None:
        row = [
            "04",
            "Tỉnh Cao Bằng",
            "042",
            "Huyện Bảo Lâm",
            "01304",
            "Xã Thạch Lâm",
            "legacy-decision",
            "30/06/2004",
            "Xã Quảng Lâm",
            "01304",
            "current-decision",
            "01/07/2025",
            "Tỉnh Cao Bằng",
            "04",
            "note",
        ]

        record = normalize_ward_comparison_rows([row])[0]

        self.assertEqual(record["legacy_district_name"], "Huyện Bảo Lâm")
        self.assertEqual(record["legacy_ward_name"], "Xã Thạch Lâm")
        self.assertEqual(record["current_ward_name"], "Xã Quảng Lâm")
        self.assertEqual(record["current_province_code"], "04")

    def test_normalizes_province_comparison_columns(self) -> None:
        row = [
            "02",
            "Tỉnh Hà Giang",
            "legacy-decision",
            "01/01/1997",
            "Tỉnh Tuyên Quang",
            "08",
            "current-decision",
            "01/07/2025",
            "merge note",
        ]

        record = normalize_province_comparison_rows([row])[0]

        self.assertEqual(record["legacy_province_code"], "02")
        self.assertEqual(record["current_province_code"], "08")

    def test_reports_direct_and_unmatched_rows(self) -> None:
        ward_rows = [
            {
                "legacy_ward_code": "00001",
                "current_ward_code": "",
            },
            {
                "legacy_ward_code": "01304",
                "current_ward_code": "01304",
            },
            {
                "legacy_ward_code": "",
                "current_ward_code": "00004",
            },
        ]
        province_rows = [{"legacy_province_code": "01"}]

        report = build_report(ward_rows, province_rows)

        self.assertEqual(report["direct_ward_mappings"], 1)
        self.assertEqual(report["legacy_only_rows"], 1)
        self.assertEqual(report["current_only_rows"], 1)


if __name__ == "__main__":
    unittest.main()
