# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import torch
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
    assert preprocessed_tokens[2] == ner_test_data["EXPECTED_TRAILING_TOKEN_MASK"]
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
    assert preprocessed_tokens[2] == ner_test_data["EXPECTED_TRAILING_TOKEN_MASK"]
    assert preprocessed_tokens[3] == ner_test_data["EXPECTED_LABEL_IDS"]

    # test not providing labels
    preprocessed_tokens = bert_english_tokenizer.tokenize_ner(
        text=ner_test_data["INPUT_TEXT"], label_map=ner_test_data["LABEL_MAP"], max_len=20
    )
    assert preprocessed_tokens[2] == ner_test_data["EXPECTED_TRAILING_TOKEN_MASK"]

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


## Tests for tokenize_qa method
@pytest.fixture()
def qa_test_data(tmp_path):

    return {
        "max_seq_len": 64,
        "doc_stride": 32,
        "max_query_len": 13,
        "cached_features_file": os.path.join(tmp_path, "cached_features_train"),
        "cached_examples_file": os.path.join(tmp_path, "cached_examples_train"),
        "doc_text": [
            "The color of the sky is blue.",
            "Architecturally, the school has a Catholic character. Atop the Main Building's "
            "gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main "
            "Building and facing it, is a copper statue of Christ with arms upraised with the "
            'legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the '
            "Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of "
            "prayer and reflection. It is a replica of the grotto at Lourdes, France where "
            "the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At "
            "the end of the main drive (and in a direct line that connects through 3 statues "
            "and the Gold Dome), is a simple, modern stone statue of Mary.",
        ],
        "question_text": [
            "What's the color of the sky?",
            "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
        ],
        "answer_start": [24, 515],
        "answer_text": ["blue", "Saint Bernadette Soubirous"],
        "qa_id": [1, 2],
        "expected_feature_count": 6,
        "expected_example_count": 2,
        "answer_start_list": [[24], [515]],
        "answer_text_list": [["blue"], ["Saint Bernadette Soubirous"]],
        "answer_start_multiple": [[24, 25], [515, 521]],
        "answer_text_multiple": [
            ["blue", "lue"],
            ["Saint Bernadette Soubirous", "Bernadette Soubirous"],
        ],
        "doc_text_with_impossible": [
            "The color of the sky is blue.",
            "Architecturally, the school has a Catholic character. Atop the Main Building's "
            "gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main "
            "Building and facing it, is a copper statue of Christ with arms upraised with the "
            'legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the '
            "Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of "
            "prayer and reflection. It is a replica of the grotto at Lourdes, France where "
            "the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At "
            "the end of the main drive (and in a direct line that connects through 3 statues "
            "and the Gold Dome), is a simple, modern stone statue of Mary.",
            "Dan has a daughter.",
        ],
        "question_text_with_impossible": [
            "What's the color of the sky?",
            "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
            "How old is Dan's daughter?",
        ],
        "answer_start_with_impossible": [24, 515, -1],
        "answer_text_with_impossible": ["blue", "Saint Bernadette Soubirous", ""],
        "is_impossible": [False, False, True],
        "qa_id_with_impossible": [1, 2, 3],
        "expected_feature_count_with_impossible": 7,
        "expected_example_count_with_impossible": 3,
    }


def test_tokenize_qa_is_training(qa_test_data, bert_english_tokenizer):
    qa_tokenization_result = bert_english_tokenizer.tokenize_qa(
        doc_text=qa_test_data["doc_text"],
        question_text=qa_test_data["question_text"],
        is_training=True,
        qa_id=qa_test_data["qa_id"],
        answer_start=qa_test_data["answer_start"],
        answer_text=qa_test_data["answer_text"],
        max_len=qa_test_data["max_seq_len"],
        doc_stride=qa_test_data["doc_stride"],
        max_question_length=qa_test_data["max_query_len"],
    )

    features = qa_tokenization_result[0]
    examples = qa_tokenization_result[1]
    max_seq_len = qa_test_data["max_seq_len"]

    assert len(features) == qa_test_data["expected_feature_count"]
    assert len(examples) == qa_test_data["expected_example_count"]

    for f in features:
        assert len(f.input_ids) == max_seq_len
        assert len(f.segment_ids) == max_seq_len
        assert len(f.input_mask) == max_seq_len


def test_tokenize_qa_is_training_answer_list(qa_test_data, bert_english_tokenizer):
    qa_tokenization_result = bert_english_tokenizer.tokenize_qa(
        doc_text=qa_test_data["doc_text"],
        question_text=qa_test_data["question_text"],
        is_training=True,
        qa_id=qa_test_data["qa_id"],
        answer_start=qa_test_data["answer_start_list"],
        answer_text=qa_test_data["answer_text_list"],
        max_len=qa_test_data["max_seq_len"],
        doc_stride=qa_test_data["doc_stride"],
    )

    features = qa_tokenization_result[0]
    examples = qa_tokenization_result[1]
    max_seq_len = qa_test_data["max_seq_len"]

    assert len(features) == qa_test_data["expected_feature_count"]
    assert len(examples) == qa_test_data["expected_example_count"]

    for f in features:
        assert len(f.input_ids) == max_seq_len
        assert len(f.segment_ids) == max_seq_len
        assert len(f.input_mask) == max_seq_len


def test_tokenize_qa_is_training_no_answer(qa_test_data, bert_english_tokenizer):
    with pytest.raises(Exception):
        bert_english_tokenizer.tokenize_qa(
            doc_text=qa_test_data["doc_text"],
            question_text=qa_test_data["question_text"],
            is_training=True,
            qa_id=qa_test_data["qa_id"],
            max_len=qa_test_data["max_seq_len"],
            doc_stride=qa_test_data["doc_stride"],
        )


def test_tokenize_qa_is_training_multiple_answers(qa_test_data, bert_english_tokenizer):
    with pytest.raises(Exception):
        bert_english_tokenizer.tokenize_qa(
            doc_text=qa_test_data["doc_text"],
            question_text=qa_test_data["question_text"],
            is_training=True,
            qa_id=qa_test_data["qa_id"],
            answer_start=qa_test_data["answer_start_multiple"],
            answer_text=qa_test_data["answer_text_multiple"],
            max_len=qa_test_data["max_seq_len"],
            doc_stride=qa_test_data["doc_stride"],
        )


def test_tokenize_qa_is_training_is_impossible(qa_test_data, bert_english_tokenizer):
    qa_tokenization_result = bert_english_tokenizer.tokenize_qa(
        doc_text=qa_test_data["doc_text_with_impossible"],
        question_text=qa_test_data["question_text_with_impossible"],
        is_training=True,
        qa_id=qa_test_data["qa_id_with_impossible"],
        answer_start=qa_test_data["answer_start_with_impossible"],
        answer_text=qa_test_data["answer_text_with_impossible"],
        max_len=qa_test_data["max_seq_len"],
        doc_stride=qa_test_data["doc_stride"],
        is_impossible=qa_test_data["is_impossible"],
    )

    features = qa_tokenization_result[0]
    examples = qa_tokenization_result[1]
    max_seq_len = qa_test_data["max_seq_len"]

    assert len(features) == qa_test_data["expected_feature_count_with_impossible"]
    assert len(examples) == qa_test_data["expected_example_count_with_impossible"]

    for f in features:
        assert len(f.input_ids) == max_seq_len
        assert len(f.segment_ids) == max_seq_len
        assert len(f.input_mask) == max_seq_len
    assert features[-1].start_position == 0
    assert features[-1].end_position == 0


def test_tokenize_qa_is_training_False(qa_test_data, bert_english_tokenizer):
    qa_tokenization_result = bert_english_tokenizer.tokenize_qa(
        doc_text=qa_test_data["doc_text"],
        question_text=qa_test_data["question_text"],
        is_training=False,
        qa_id=qa_test_data["qa_id"],
        max_len=qa_test_data["max_seq_len"],
        doc_stride=qa_test_data["doc_stride"],
    )

    features = qa_tokenization_result[0]
    examples = qa_tokenization_result[1]
    max_seq_len = qa_test_data["max_seq_len"]

    assert len(features) == qa_test_data["expected_feature_count"]
    assert len(examples) == qa_test_data["expected_example_count"]

    for f in features:
        assert len(f.input_ids) == max_seq_len
        assert len(f.segment_ids) == max_seq_len
        assert len(f.input_mask) == max_seq_len


def test_tokenize_qa_cache_result(qa_test_data, bert_english_tokenizer):
    qa_tokenization_result = bert_english_tokenizer.tokenize_qa(
        doc_text=qa_test_data["doc_text"],
        question_text=qa_test_data["question_text"],
        is_training=True,
        qa_id=qa_test_data["qa_id"],
        answer_start=qa_test_data["answer_start"],
        answer_text=qa_test_data["answer_text"],
        max_len=qa_test_data["max_seq_len"],
        doc_stride=qa_test_data["doc_stride"],
        max_question_length=qa_test_data["max_query_len"],
        cache_results=True,
    )

    features = qa_tokenization_result[0]
    examples = qa_tokenization_result[1]

    assert features == torch.load(qa_test_data["cached_features_file"])
    assert examples == torch.load(qa_test_data["cached_examples_file"])


## Tests for tokenize_qa method end
