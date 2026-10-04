import unittest

from address_ner.split_dataset import split_by_group


class SplitDatasetTests(unittest.TestCase):
    def test_keeps_groups_in_one_split(self) -> None:
        records = [
            {"id": f"item_{index}", "group_id": f"group_{index // 2}"}
            for index in range(12)
        ]

        train, validation, test = split_by_group(
            records,
            validation_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        group_sets = [
            {record["group_id"] for record in split}
            for split in (train, validation, test)
        ]
        self.assertTrue(group_sets[0].isdisjoint(group_sets[1]))
        self.assertTrue(group_sets[0].isdisjoint(group_sets[2]))
        self.assertTrue(group_sets[1].isdisjoint(group_sets[2]))
        self.assertEqual(sum(map(len, (train, validation, test))), len(records))

    def test_stratification_places_every_kind_in_every_split(self) -> None:
        records = [
            {
                "id": f"{kind}_{index}",
                "group_id": f"{kind}_{index}",
                "generation": {"kind": kind},
            }
            for kind in ("current", "legacy")
            for index in range(5)
        ]

        splits = split_by_group(
            records,
            validation_ratio=0.2,
            test_ratio=0.2,
            seed=42,
            stratify_field="generation.kind",
        )

        for split in splits:
            self.assertEqual(
                {record["generation"]["kind"] for record in split},
                {"current", "legacy"},
            )


if __name__ == "__main__":
    unittest.main()
