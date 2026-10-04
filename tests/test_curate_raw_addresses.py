import tempfile
import unittest
import unicodedata
from pathlib import Path

from address_ner.curate_raw_addresses import (
    classify_signals,
    curate_addresses,
    normalize_address,
    read_raw_addresses,
)


class CurateRawAddressesTests(unittest.TestCase):
    def test_normalizes_unicode_whitespace_and_zero_width(self) -> None:
        decomposed = unicodedata.normalize("NFD", "Hà Nội")
        self.assertEqual(normalize_address(f"  {decomposed}\u200b   "), "Hà Nội")

    def test_deduplicates_case_and_whitespace_only(self) -> None:
        accepted, rejected = curate_addresses(
            [
                (1, "12 Nguyễn Trãi, Hà Nội"),
                (2, "  12 NGUYỄN   TRÃI, HÀ NỘI "),
                (3, "12 Nguyen Trai, Ha Noi"),
            ],
            "test",
        )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(rejected[0]["reasons"], ["duplicate"])
        self.assertEqual(rejected[0]["duplicate_of"], "curated_000001")

    def test_quarantines_contact_details(self) -> None:
        accepted, rejected = curate_addresses(
            [
                (1, "12 Nguyễn Trãi, email a@example.com"),
                (2, "12 Nguyễn Trãi, 0912 345 678"),
            ],
            "test",
        )
        self.assertEqual(accepted, [])
        self.assertIn("contains_email", rejected[0]["reasons"])
        self.assertIn("contains_phone", rejected[1]["reasons"])

    def test_classifies_sampling_signals(self) -> None:
        signals = classify_signals("12 Nguyen Trai, P. Ben Thanh, Q.1, TP. HCM")
        self.assertIn("HAS_HOUSE_NUMBER", signals)
        self.assertIn("ASCII_ONLY", signals)
        self.assertIn("ABBREVIATED", signals)
        self.assertIn("HAS_LEGACY_DISTRICT", signals)

    def test_auto_detects_csv_address_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "addresses.csv"
            path.write_text(
                "id,dia_chi\n1,12 Nguyễn Trãi\n", encoding="utf-8"
            )
            rows = read_raw_addresses(path)
        self.assertEqual(rows, [(2, "12 Nguyễn Trãi")])


if __name__ == "__main__":
    unittest.main()
