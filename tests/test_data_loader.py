import unittest

import torch

from address_ner.data_loader import AddressNerDataset, DynamicPaddingCollator
from address_ner.labels import LABEL2ID
from address_ner.tokenize_dataset import IGNORE_INDEX


class FakeTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 1

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


class DynamicPaddingTests(unittest.TestCase):
    def test_pads_to_longest_sequence_in_batch(self) -> None:
        records = [
            {
                "id": "short",
                "tokens": ["12"],
                "ner_tags": ["B-HOUSE_NUMBER"],
            },
            {
                "id": "long",
                "tokens": ["12", "Nguyễn_Trãi"],
                "ner_tags": ["B-HOUSE_NUMBER", "B-STREET"],
            },
        ]
        tokenizer = FakeTokenizer()
        dataset = AddressNerDataset(records, tokenizer, max_length=8)
        collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)

        batch = collator([dataset[0], dataset[1]])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))
        self.assertEqual(
            batch["input_ids"].tolist(),
            [[0, 10, 2, 1, 1], [0, 10, 11, 12, 2]],
        )
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
        )
        self.assertEqual(
            batch["labels"].tolist(),
            [
                [IGNORE_INDEX, LABEL2ID["B-HOUSE_NUMBER"], IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
                [
                    IGNORE_INDEX,
                    LABEL2ID["B-HOUSE_NUMBER"],
                    LABEL2ID["B-STREET"],
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                ],
            ],
        )
        self.assertEqual(batch["input_ids"].dtype, torch.long)


if __name__ == "__main__":
    unittest.main()

