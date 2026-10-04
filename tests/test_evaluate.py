import unittest

from address_ner.evaluate import compute_entity_metrics


def entity(entity_type: str, start: int, end: int) -> dict:
    return {"type": entity_type, "start_word": start, "end_word": end}


class EntityMetricsTests(unittest.TestCase):
    def test_requires_exact_type_and_span_match(self):
        results = [
            {
                "predicted_entities": [
                    entity("WARD", 2, 3),
                    entity("DISTRICT", 5, 5),
                    entity("PROVINCE", 6, 6),
                ],
                "gold_entities": [
                    entity("STREET", 0, 0),
                    entity("WARD", 2, 3),
                    entity("DISTRICT", 5, 6),
                ],
            }
        ]

        metrics = compute_entity_metrics(results)

        self.assertEqual(metrics["overall"]["tp"], 1)
        self.assertEqual(metrics["overall"]["fp"], 2)
        self.assertEqual(metrics["overall"]["fn"], 2)
        self.assertAlmostEqual(metrics["overall"]["precision"], 1 / 3)
        self.assertAlmostEqual(metrics["overall"]["recall"], 1 / 3)
        self.assertAlmostEqual(metrics["overall"]["f1"], 1 / 3)
        self.assertEqual(metrics["per_type"]["WARD"]["tp"], 1)
        self.assertEqual(metrics["per_type"]["DISTRICT"]["fp"], 1)
        self.assertEqual(metrics["per_type"]["DISTRICT"]["fn"], 1)

    def test_returns_zero_when_no_entities_exist(self):
        metrics = compute_entity_metrics(
            [{"predicted_entities": [], "gold_entities": []}]
        )

        self.assertEqual(metrics["overall"]["precision"], 0.0)
        self.assertEqual(metrics["overall"]["recall"], 0.0)
        self.assertEqual(metrics["overall"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
