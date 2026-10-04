import unittest

import torch

from address_ner.inference import decode_word_predictions, merge_bio_entities
from address_ner.labels import LABEL2ID


def logits_for(labels: list[str]) -> torch.Tensor:
    logits = torch.full((len(labels), len(LABEL2ID)), -10.0)
    for index, label in enumerate(labels):
        logits[index, LABEL2ID[label]] = 10.0
    return logits


class InferenceTests(unittest.TestCase):
    def test_uses_only_first_subword_prediction(self) -> None:
        words = ["12", "Nguyễn_Trãi"]
        word_ids = [None, 0, 1, 1, None]
        logits = logits_for(["O", "B-HOUSE_NUMBER", "B-STREET", "B-WARD", "O"])

        predictions = decode_word_predictions(words, word_ids, logits)

        self.assertEqual(
            [prediction["label"] for prediction in predictions],
            ["B-HOUSE_NUMBER", "B-STREET"],
        )

    def test_merges_bio_entity(self) -> None:
        predictions = [
            {"word_index": 0, "token": "Phường", "label": "B-WARD", "confidence": 0.9},
            {"word_index": 1, "token": "Bến_Thành", "label": "I-WARD", "confidence": 0.8},
        ]

        entities = merge_bio_entities(predictions)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["type"], "WARD")
        self.assertEqual(entities[0]["text"], "Phường Bến Thành")
        self.assertFalse(entities[0]["bio_repaired"])
        self.assertAlmostEqual(entities[0]["mean_confidence"], 0.85)

    def test_repairs_invalid_i_start(self) -> None:
        predictions = [
            {"word_index": 0, "token": "Bến_Thành", "label": "I-WARD", "confidence": 0.7}
        ]

        entities = merge_bio_entities(predictions)

        self.assertEqual(entities[0]["type"], "WARD")
        self.assertTrue(entities[0]["bio_repaired"])

    def test_joins_split_district_prefix_and_name(self) -> None:
        predictions = [
            {
                "word_index": 0,
                "token": "Huyện",
                "label": "B-DISTRICT",
                "confidence": 0.7,
            },
            {
                "word_index": 1,
                "token": "Bảo_Lâm",
                "label": "I-WARD",
                "confidence": 0.8,
            },
        ]

        entities = merge_bio_entities(predictions)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["type"], "DISTRICT")
        self.assertEqual(entities[0]["text"], "Huyện Bảo Lâm")
        self.assertEqual(
            entities[0]["structure_repaired"],
            "split_district_name",
        )
        self.assertAlmostEqual(entities[0]["mean_confidence"], 0.75)


if __name__ == "__main__":
    unittest.main()
