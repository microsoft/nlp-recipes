# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from utils_nlp.models.bert.token_classification import (
    BERTTokenClassifier,
    postprocess_token_labels,
)


def test_token_classifier_num_labels():
    with pytest.raises(ValueError):
        BERTTokenClassifier(num_labels=1)


def test_token_classifier_fit_predict(tmp_path, ner_test_data):
    token_classifier = BERTTokenClassifier(num_labels=6, cache_dir=tmp_path)

    # test fit, no warmup
    token_classifier.fit(
        token_ids=ner_test_data["INPUT_TOKEN_IDS"],
        input_mask=ner_test_data["INPUT_MASK"],
        labels=ner_test_data["INPUT_LABEL_IDS"],
    )

    # test fit, with warmup
    token_classifier.fit(
        token_ids=ner_test_data["INPUT_TOKEN_IDS"],
        input_mask=ner_test_data["INPUT_MASK"],
        labels=ner_test_data["INPUT_LABEL_IDS"],
        warmup_proportion=0.1,
    )
    # test predict, no labels
    token_classifier.predict(
        token_ids=ner_test_data["INPUT_TOKEN_IDS"],
        input_mask=ner_test_data["INPUT_MASK"],
    )

    # test predict, with labels
    token_classifier.predict(
        token_ids=ner_test_data["INPUT_TOKEN_IDS"],
        input_mask=ner_test_data["INPUT_MASK"],
        labels=ner_test_data["INPUT_LABEL_IDS"],
    )

    # test output probabilities
    predictions = token_classifier.predict(
        token_ids=ner_test_data["INPUT_TOKEN_IDS"],
        input_mask=ner_test_data["INPUT_MASK"],
        labels=ner_test_data["INPUT_LABEL_IDS"],
        probabilities=True,
    )
    assert len(predictions.classes) == predictions.probabilities.shape[0]


def test_postprocess_token_labels(ner_test_data):
    labels_no_padding = postprocess_token_labels(
        labels=ner_test_data["PREDICTED_LABELS"],
        input_mask=ner_test_data["INPUT_MASK"],
        label_map=ner_test_data["LABEL_MAP"],
    )

    assert labels_no_padding == ner_test_data["EXPECTED_TOKENS_NO_PADDING"]


def test_postprocess_token_labels_remove_trailing(ner_test_data):
    labels_no_padding_no_trailing = postprocess_token_labels(
        labels=ner_test_data["PREDICTED_LABELS"],
        input_mask=ner_test_data["INPUT_MASK"],
        label_map=ner_test_data["LABEL_MAP"],
        remove_trailing_word_pieces=True,
        trailing_token_mask=ner_test_data["TRAILING_TOKEN_MASK"],
    )

    assert (
        labels_no_padding_no_trailing
        == ner_test_data["EXPECTED_TOKENS_NO_PADDING_NO_TRAILING"]
    )
