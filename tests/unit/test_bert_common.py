# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from utils_nlp.models.bert.common import create_data_loader


def test_tokenize(bert_english_tokenizer):
    text = ["Hello World.", "How you doing?", "greatttt"]
    tokens = bert_english_tokenizer.tokenize(text)
    assert len(tokens) == len(text)
    assert len(tokens[0]) == 3
    assert len(tokens[1]) == 4
    assert len(tokens[2]) == 3
    assert tokens[2][1].startswith("##")


def test_tokenize_ner(ner_test_data, bert_english_tokenizer):
    seq_length = 20

    # test providing labels
    preprocessed_tokens = bert_english_tokenizer.tokenize_ner(
        text=ner_test_data["INPUT_TEXT"],
        labels=ner_test_data["INPUT_LABELS"],
        label_map=ner_test_data["LABEL_MAP"],
        max_len=seq_length,
    )

    assert len(preprocessed_tokens[0][0]) == seq_length
    assert len(preprocessed_tokens[1][0]) == seq_length
    assert (
        preprocessed_tokens[2] == ner_test_data["EXPECTED_TRAILING_TOKEN_MASK"]
    )
    assert preprocessed_tokens[3] == ner_test_data["EXPECTED_LABEL_IDS"]

    # test when input is a single list
    preprocessed_tokens = bert_english_tokenizer.tokenize_ner(
        text=ner_test_data["INPUT_TEXT_SINGLE"],
        labels=ner_test_data["INPUT_LABELS_SINGLE"],
        label_map=ner_test_data["LABEL_MAP"],
        max_len=seq_length,
    )

    assert len(preprocessed_tokens[0][0]) == seq_length
    assert len(preprocessed_tokens[1][0]) == seq_length
    assert (
        preprocessed_tokens[2] == ner_test_data["EXPECTED_TRAILING_TOKEN_MASK"]
    )
    assert preprocessed_tokens[3] == ner_test_data["EXPECTED_LABEL_IDS"]

    # test not providing labels
    preprocessed_tokens = bert_english_tokenizer.tokenize_ner(
        text=ner_test_data["INPUT_TEXT"],
        label_map=ner_test_data["LABEL_MAP"],
        max_len=20,
    )
    assert (
        preprocessed_tokens[2] == ner_test_data["EXPECTED_TRAILING_TOKEN_MASK"]
    )

    # text exception when number of words and number of labels are different
    with pytest.raises(ValueError):
        preprocessed_tokens = bert_english_tokenizer.tokenize_ner(
            text=ner_test_data["INPUT_TEXT"],
            labels=ner_test_data["INPUT_LABELS_WRONG"],
            label_map=ner_test_data["LABEL_MAP"],
            max_len=seq_length,
        )


def test_create_data_loader(ner_test_data):
    with pytest.raises(ValueError):
        create_data_loader(
            input_ids=ner_test_data["INPUT_TOKEN_IDS"],
            input_mask=ner_test_data["INPUT_MASK"],
            label_ids=ner_test_data["INPUT_LABEL_IDS"],
            sample_method="dummy",
        )

    create_data_loader(
        input_ids=ner_test_data["INPUT_TOKEN_IDS"],
        input_mask=ner_test_data["INPUT_MASK"],
        label_ids=ner_test_data["INPUT_LABEL_IDS"],
        sample_method="sequential",
    )

    create_data_loader(
        input_ids=ner_test_data["INPUT_TOKEN_IDS"],
        input_mask=ner_test_data["INPUT_MASK"],
        label_ids=ner_test_data["INPUT_LABEL_IDS"],
        sample_method="random",
    )
