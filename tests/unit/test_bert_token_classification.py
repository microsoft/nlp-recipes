# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import shutil
import pytest

nlp_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.bert.token_classification import (
    BERTTokenClassifier,
    postprocess_token_labels,
)

# Test data
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
    [3, 5, 5, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
INPUT_MASK = [[1] * 11 + [0] * 9]
PREDICTED_LABELS = [
    [3, 5, 5, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
TRAILING_TOKEN_MASK = [[True] * 20]
false_pos = [1, 2]
for p in false_pos:
    TRAILING_TOKEN_MASK[0][p] = False

UNIQUE_LABELS = ["O", "I-LOC", "I-MISC", "I-PER", "I-ORG", "X"]
LABEL_MAP = {label: i for i, label in enumerate(UNIQUE_LABELS)}

CACHE_DIR = "./test_bert_token_cache"


def test_token_classifier_num_labels():
    with pytest.raises(ValueError):
        BERTTokenClassifier(num_labels=1)


def test_token_classifier_fit_predict():
    token_classifier = BERTTokenClassifier(num_labels=6, cache_dir=CACHE_DIR)

    # test fit, no warmup
    token_classifier.fit(
        token_ids=INPUT_TOKEN_IDS,
        input_mask=INPUT_MASK,
        labels=INPUT_LABEL_IDS,
    )

    # test fit, with warmup
    token_classifier.fit(
        token_ids=INPUT_TOKEN_IDS,
        input_mask=INPUT_MASK,
        labels=INPUT_LABEL_IDS,
        warmup_proportion=0.1,
    )
    # test predict, no labels
    token_classifier.predict(token_ids=INPUT_TOKEN_IDS, input_mask=INPUT_MASK)

    # test predict, with labels
    token_classifier.predict(
        token_ids=INPUT_TOKEN_IDS,
        input_mask=INPUT_MASK,
        labels=INPUT_LABEL_IDS,
    )
    shutil.rmtree(CACHE_DIR)


def test_postprocess_token_labels():
    expected_labels_no_padding = [
        ["I-PER", "X", "X", "O", "O", "O", "O", "I-ORG", "I-ORG", "I-ORG", "O"]
    ]

    labels_no_padding = postprocess_token_labels(
        labels=PREDICTED_LABELS, input_mask=INPUT_MASK, label_map=LABEL_MAP
    )

    assert labels_no_padding == expected_labels_no_padding


def test_postprocess_token_labels_remove_trailing():
    expected_postprocessed_labels = [
        ["I-PER", "O", "O", "O", "O", "I-ORG", "I-ORG", "I-ORG", "O"]
    ]

    labels_no_padding_no_trailing = postprocess_token_labels(
        labels=PREDICTED_LABELS,
        input_mask=INPUT_MASK,
        label_map=LABEL_MAP,
        remove_trailing_word_pieces=True,
        trailing_token_mask=TRAILING_TOKEN_MASK,
    )

    assert labels_no_padding_no_trailing == expected_postprocessed_labels
