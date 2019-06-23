# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest


from utils_nlp.bert.common import Tokenizer, create_data_loader, Language


def test_tokenizer_preprocess_ner_tokens(ner_test_data):
    seq_length = 20

    tokenizer = Tokenizer(language=Language.ENGLISHCASED, to_lower=False)

    # test providing labels
    preprocessed_tokens = tokenizer.tokenize_preprocess_ner_text(
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

    # test not providing labels
    preprocessed_tokens = tokenizer.tokenize_preprocess_ner_text(
        text=ner_test_data["INPUT_TEXT"],
        label_map=ner_test_data["LABEL_MAP"],
        max_len=20,
    )
    assert (
        preprocessed_tokens[2] == ner_test_data["EXPECTED_TRAILING_TOKEN_MASK"]
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
