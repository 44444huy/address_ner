import unittest

from address_ner.validate_annotations import validate_annotation_record


def record(text="Nguyễn Trãi, Hà Nội", labels=None, status="approved"):
    return {
        "id": "raw_000001",
        "text": text,
        "label": labels or [],
        "annotation_status": status,
        "source": "real",
    }


class AnnotationValidationTests(unittest.TestCase):
    def test_accepts_address_without_house_number(self):
        text = "Nguyễn Trãi, Hà Nội"
        start = text.index("Hà Nội")
        issues = validate_annotation_record(
            record(text, [[start, start + len("Hà Nội"), "PROVINCE"]])
        )

        self.assertEqual(issues, [])

    def test_rejects_overlapping_spans(self):
        issues = validate_annotation_record(
            record("abcdef", [[0, 4, "STREET"], [3, 6, "WARD"]])
        )

        self.assertTrue(any("overlap" in issue.message for issue in issues))

    def test_rejects_out_of_bounds_span(self):
        issues = validate_annotation_record(
            record("abc", [[0, 10, "PROVINCE"]])
        )

        self.assertTrue(any("invalid range" in issue.message for issue in issues))

    def test_rejects_whitespace_inside_span_boundaries(self):
        issues = validate_annotation_record(
            record(" Hà Nội ", [[0, 7, "PROVINCE"]])
        )

        self.assertTrue(any("whitespace" in issue.message for issue in issues))


if __name__ == "__main__":
    unittest.main()
