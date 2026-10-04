import unittest

from address_ner.prepare_annotation import (
    AdminUnit,
    CatalogMatcher,
    create_annotation_tasks,
    iter_components,
)


def unit(province_code, province_name, ward_code, ward_name, unit_type, short_name):
    return AdminUnit(
        province_code=province_code,
        province_name=province_name,
        ward_code=ward_code,
        ward_name=ward_name,
        unit_type=unit_type,
        short_name=short_name,
    )


class PrepareAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.matcher = CatalogMatcher(
            [
                unit("01", "Thành phố Hà Nội", "00004", "Phường Ba Đình", "PHƯỜNG", "Ba Đình"),
                unit("79", "Thành phố Hồ Chí Minh", "26734", "Phường Bến Thành", "PHƯỜNG", "Bến Thành"),
                unit("01", "Thành phố Hà Nội", "99901", "Phường Trung Sơn", "PHƯỜNG", "Trung Sơn"),
                unit("79", "Thành phố Hồ Chí Minh", "99979", "Phường Trung Sơn", "PHƯỜNG", "Trung Sơn"),
            ]
        )

    def test_component_offsets_point_to_original_text(self):
        text = "Nguyễn Trãi,  Phường Ba Đình ; Hà Nội"
        components = list(iter_components(text))

        self.assertEqual([component.text for component in components], ["Nguyễn Trãi", "Phường Ba Đình", "Hà Nội"])
        for component in components:
            self.assertEqual(text[component.start : component.end], component.text)

    def test_address_without_house_number_is_valid(self):
        text = "Nguyễn Trãi, Phường Ba Đình, Hà Nội"
        labels, suggestions = self.matcher.suggest(text)

        self.assertEqual([label[2] for label in labels], ["WARD", "PROVINCE"])
        self.assertNotIn("HOUSE_NUMBER", [label[2] for label in labels])
        for start, end, _ in labels:
            self.assertTrue(text[start:end])
        self.assertEqual(len(suggestions), 2)

    def test_parent_province_resolves_duplicate_short_ward_name(self):
        labels, suggestions = self.matcher.suggest("Trung Sơn, Hà Nội")

        self.assertEqual([label[2] for label in labels], ["WARD", "PROVINCE"])
        ward = next(item for item in suggestions if item["label"] == "WARD")
        self.assertEqual(ward["code"], "99901")

    def test_ambiguous_short_ward_without_province_is_not_suggested(self):
        labels, suggestions = self.matcher.suggest("Trung Sơn")

        self.assertEqual(labels, [])
        self.assertEqual(suggestions, [])

    def test_every_task_still_requires_human_review(self):
        tasks = create_annotation_tasks(
            ["", "Phường Bến Thành, TP. Hồ Chí Minh"],
            self.matcher,
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["id"], "raw_000002")
        self.assertEqual(tasks[0]["annotation_status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
