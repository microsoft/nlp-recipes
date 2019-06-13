# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import pytest

nlp_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.bert.common import Tokenizer, create_data_loader, Language

INPUT_TEXT = ["Johnathan is studying in the University of Michigan."]
INPUT_LABELS = [["I-PER", "O", "O", "O", "O", "I-ORG", "I-ORG", "I-ORG"]]
INPUT_TOKEN_IDS = [
    [
        1287,
        9779,
        1389,
        1110,
        5076,
        1107,
        1103,
        1239,
        1104,
        3312,
        119,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
]
INPUT_LABEL_IDS = [
    [3, 5, 5, 0, 0, 0, 0, 4, 4, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
INPUT_MASK = [[1] * 11 + [0] * 9]


UNIQUE_LABELS = ["O", "I-LOC", "I-MISC", "I-PER", "I-ORG", "X"]
LABEL_MAP = {label: i for i, label in enumerate(UNIQUE_LABELS)}


def test_tokenizer_preprocess_ner_tokens():
    expected_trailing_token_mask = [[True] * 20]
    false_pos = [1, 2, 10]
    for p in false_pos:
        expected_trailing_token_mask[0][p] = False
    expected_label_ids = [
        [3, 5, 5, 0, 0, 0, 0, 4, 4, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    seq_length = 20

    tokenizer = Tokenizer(language=Language.ENGLISHCASED, to_lower=False)

    # test providing labels
    preprocessed_tokens = tokenizer.preprocess_ner_tokens(
        text=INPUT_TEXT,
        labels=INPUT_LABELS,
        label_map=LABEL_MAP,
        max_len=seq_length,
    )

    assert len(preprocessed_tokens[0][0]) == seq_length
    assert len(preprocessed_tokens[1][0]) == seq_length
    assert preprocessed_tokens[2] == expected_trailing_token_mask
    assert preprocessed_tokens[3] == expected_label_ids

    # test not providing labels
    preprocessed_tokens = tokenizer.preprocess_ner_tokens(
        text=INPUT_TEXT, label_map=LABEL_MAP, max_len=20
    )
    assert preprocessed_tokens[2] == expected_trailing_token_mask


def test_create_data_loader():
    with pytest.raises(ValueError):
        create_data_loader(
            input_ids=INPUT_TOKEN_IDS,
            input_mask=INPUT_MASK,
            label_ids=INPUT_LABEL_IDS,
            sample_method="dummy",
        )

    create_data_loader(
        input_ids=INPUT_TOKEN_IDS,
        input_mask=INPUT_MASK,
        label_ids=INPUT_LABEL_IDS,
        sample_method="sequential",
    )

    create_data_loader(
        input_ids=INPUT_TOKEN_IDS,
        input_mask=INPUT_MASK,
        label_ids=INPUT_LABEL_IDS,
        sample_method="random",
    )
