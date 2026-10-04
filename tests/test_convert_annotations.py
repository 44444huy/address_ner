import unittest

from address_ner.convert_annotations import (
    align_char_spans_to_bio,
    convert_record,
    locate_segmented_tokens,
    split_address_chunks,
)


class FixedSegmenter:
    def __init__(self, tokens):
        self.tokens = tokens

    def segment(self, text):
        return self.tokens


class ConvertAnnotationsTests(unittest.TestCase):
    def test_splits_before_administrative_markers(self):
        self.assertEqual(
            split_address_chunks("12 Nguyễn Trãi Phường Ba Đình Hà Nội"),
            ["12 Nguyễn Trãi", "Phường Ba Đình Hà Nội"],
        )
        self.assertEqual(
            split_address_chunks("Thị xã Việt Yên, Tỉnh Bắc Ninh"),
            ["Thị xã Việt Yên,", "Tỉnh Bắc Ninh"],
        )
        self.assertEqual(
            split_address_chunks("12A/5, Đường Trường Chinh, Phường Ba Đình"),
            ["12A/5,", "Đường", "Trường Chinh,", "Phường Ba Đình"],
        )

    def test_maps_underscored_tokens_to_character_offsets(self):
        text = "Nguyễn Trãi, Hà Nội"
        spans = locate_segmented_tokens(
            text, ["Nguyễn_Trãi", ",", "Hà_Nội"]
        )

        self.assertEqual(
            [(span.token, span.start, span.end) for span in spans],
            [("Nguyễn_Trãi", 0, 11), (",", 11, 12), ("Hà_Nội", 13, 19)],
        )

    def test_maps_equivalent_vietnamese_tone_placements(self):
        text = "Xã Đàm Thủy, Tỉnh Cao Bằng"
        spans = locate_segmented_tokens(
            text, ["Xã", "Đàm_Thuỷ", ",", "Tỉnh", "Cao_Bằng"]
        )

        self.assertEqual(spans[1].token, "Đàm_Thuỷ")
        self.assertEqual(text[spans[1].start : spans[1].end], "Đàm Thủy")

    def test_converts_approved_spans_to_bio(self):
        text = "Nguyễn Trãi, Phường Ba Đình, Hà Nội"
        ward_start = text.index("Phường")
        province_start = text.index("Hà Nội")
        record = {
            "id": "raw_000001",
            "text": text,
            "label": [
                [ward_start, ward_start + len("Phường Ba Đình"), "WARD"],
                [province_start, province_start + len("Hà Nội"), "PROVINCE"],
            ],
            "annotation_status": "approved",
            "source": "real",
        }
        segmenter = FixedSegmenter(
            ["Nguyễn_Trãi", ",", "Phường", "Ba_Đình", ",", "Hà_Nội"]
        )

        converted = convert_record(record, segmenter)

        self.assertEqual(
            converted["ner_tags"],
            ["O", "O", "B-WARD", "I-WARD", "O", "B-PROVINCE"],
        )
        self.assertNotIn("B-HOUSE_NUMBER", converted["ner_tags"])

    def test_rejects_entity_that_cuts_through_one_token(self):
        text = "Bến Thành"
        token_spans = locate_segmented_tokens(text, ["Bến_Thành"])

        with self.assertRaisesRegex(ValueError, "cuts through token"):
            align_char_spans_to_bio(
                text, token_spans, [[0, len("Bến"), "WARD"]]
            )

    def test_rejects_unreviewed_record(self):
        record = {
            "id": "raw_000001",
            "text": "Hà Nội",
            "label": [[0, 6, "PROVINCE"]],
            "annotation_status": "needs_review",
            "source": "real",
        }

        with self.assertRaisesRegex(ValueError, "not 'approved'"):
            convert_record(record, FixedSegmenter(["Hà_Nội"]))


if __name__ == "__main__":
    unittest.main()
