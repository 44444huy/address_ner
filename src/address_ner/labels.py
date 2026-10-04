"""Stable label definitions shared by training and inference."""

LABELS = [
    "O",
    "B-HOUSE_NUMBER",
    "I-HOUSE_NUMBER",
    "B-STREET",
    "I-STREET",
    "B-WARD",
    "I-WARD",
    "B-DISTRICT",
    "I-DISTRICT",
    "B-PROVINCE",
    "I-PROVINCE",
]

LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for index, label in enumerate(LABELS)}

ENTITY_TYPES = {
    "HOUSE_NUMBER",
    "STREET",
    "WARD",
    "DISTRICT",
    "PROVINCE",
}

