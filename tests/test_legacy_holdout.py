import unittest

from address_ner.build_legacy_mappings import LegacyMapping
from address_ner.legacy_holdout import (
    finalize_review_cases,
    mapping_stratum,
    select_holdout_records,
)
from address_ner.legacy_resolver import LegacyComparison


def comparison(code: str) -> LegacyComparison:
    return LegacyComparison(
        legacy_province_code="01",
        legacy_province_name="Tỉnh Cũ",
        legacy_district_code="001",
        legacy_district_name="Huyện Cũ",
        legacy_ward_code=code,
        legacy_ward_name=f"Xã {code}",
        legacy_effective_from="",
        current_ward_code="",
        current_ward_name="",
        current_province_code="",
        current_province_name="",
        current_effective_from="",
        note="",
        source_url="official",
    )


def mapping(code: str, method: str, target: str | None = None) -> LegacyMapping:
    return LegacyMapping(
        legacy_province_code="01",
        legacy_district_code="001",
        legacy_ward_code=code,
        legacy_ward_name=f"Xã {code}",
        current_province_code="91",
        current_province_name="Tỉnh Mới",
        current_ward_code=target or f"9{code}",
        current_ward_name="Xã Mới",
        mapping_method=method,
        evidence="official",
        source_url="official",
    )


class LegacyHoldoutTests(unittest.TestCase):
    def test_classifies_sampling_strata(self) -> None:
        self.assertEqual(mapping_stratum([mapping("1", "DIRECT_CODE")]), "direct")
        self.assertEqual(mapping_stratum([mapping("2", "NOTE_FULL")]), "note_full")
        self.assertEqual(mapping_stratum([mapping("3", "NOTE_PARTIAL")]), "split")
        self.assertEqual(
            mapping_stratum([mapping("4", "SOURCE_NOTE_HINT")]),
            "source_hint",
        )
        self.assertEqual(mapping_stratum([]), "unresolved")

    def test_sampling_is_balanced_deterministic_and_excludes_seed(self) -> None:
        records = [comparison(str(index)) for index in range(1, 12)]
        mappings = [
            mapping("1", "DIRECT_CODE"),
            mapping("2", "DIRECT_CODE"),
            mapping("3", "DIRECT_CODE"),
            mapping("4", "NOTE_FULL"),
            mapping("5", "NOTE_FULL"),
            mapping("6", "NOTE_PARTIAL"),
            mapping("7", "NOTE_PARTIAL"),
            mapping("8", "SOURCE_NOTE_HINT"),
            mapping("9", "SOURCE_NOTE_HINT"),
        ]

        first = select_holdout_records(records, mappings, {"1"}, 1, 42)
        second = select_holdout_records(records, mappings, {"1"}, 1, 42)

        self.assertEqual(first, second)
        self.assertEqual({stratum for stratum, _ in first}, {
            "direct", "note_full", "split", "source_hint", "unresolved"
        })
        self.assertNotIn("1", {record.legacy_ward_code for _, record in first})

    def test_finalizes_completed_review(self) -> None:
        cases = [
            {
                "id": "holdout_001",
                "review": {
                    "status": "WARNING",
                    "current_ward_code": "90001",
                    "candidate_ward_codes": ["90001"],
                    "reviewer_note": "whole-unit merge",
                },
            }
        ]
        manifest = [
            {
                "id": "holdout_001",
                "legacy_ward_code": "00001",
                "stratum": "note_full",
            }
        ]

        finalized = finalize_review_cases(cases, manifest)

        self.assertEqual(finalized[0]["expected"]["status"], "WARNING")
        self.assertEqual(
            finalized[0]["categories"], ["holdout", "note_full"]
        )

    def test_rejects_pending_review(self) -> None:
        with self.assertRaisesRegex(ValueError, "has not been reviewed"):
            finalize_review_cases(
                [{"id": "holdout_001", "review": {"status": None}}],
                [
                    {
                        "id": "holdout_001",
                        "legacy_ward_code": "00001",
                        "stratum": "unresolved",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
