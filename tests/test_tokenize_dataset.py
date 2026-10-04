import unittest

from address_ner.labels import LABEL2ID
from address_ner.tokenize_dataset import IGNORE_INDEX, tokenize_and_align_record


class FakeTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    bos_token_id = 0
    eos_token_id = 2

    token_ids = {
        "12": 10,
        "Nguyễn_@@": 11,
        "Trãi": 12,
    }

    def tokenize(self, text: str) -> list[str]:
        if text == "Nguyễn_Trãi":
            return ["Nguyễn_@@", "Trãi"]
        return [text]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.token_ids[token] for token in tokens]


class TokenizeAndAlignTests(unittest.TestCase):
    def test_aligns_only_the_first_subword(self) -> None:
        record = {
            "id": "addr_test",
            "tokens": ["12", "Nguyễn_Trãi"],
            "ner_tags": ["B-HOUSE_NUMBER", "B-STREET"],
        }

        features = tokenize_and_align_record(record, FakeTokenizer(), max_length=8)

        self.assertEqual(features["tokens"], ["<s>", "12", "Nguyễn_@@", "Trãi", "</s>"])
        self.assertEqual(features["word_ids"], [None, 0, 1, 1, None])
        self.assertEqual(
            features["labels"],
            [
                IGNORE_INDEX,
                LABEL2ID["B-HOUSE_NUMBER"],
                LABEL2ID["B-STREET"],
                IGNORE_INDEX,
                IGNORE_INDEX,
            ],
        )
        self.assertEqual(features["attention_mask"], [1, 1, 1, 1, 1])

    def test_rejects_records_over_max_length(self) -> None:
        record = {
            "id": "addr_test",
            "tokens": ["12", "Nguyễn_Trãi"],
            "ner_tags": ["B-HOUSE_NUMBER", "B-STREET"],
        }

        with self.assertRaisesRegex(ValueError, "exceeding max_length"):
            tokenize_and_align_record(record, FakeTokenizer(), max_length=4)


if __name__ == "__main__":
    unittest.main()

