# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from utils_nlp.models.bert.question_answering import BERTQAExtractor


@pytest.fixture()
def qa_test_features_examples(bert_english_tokenizer):

    raw_data = {
        "max_seq_len": 64,
        "doc_stride": 32,
        "max_query_len": 13,
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
    }

    features, examples = bert_english_tokenizer.tokenize_qa(
        doc_text=raw_data["doc_text"],
        question_text=raw_data["question_text"],
        is_training=True,
        qa_id=raw_data["qa_id"],
        answer_start=raw_data["answer_start"],
        answer_text=raw_data["answer_text"],
        max_len=raw_data["max_seq_len"],
        doc_stride=raw_data["doc_stride"],
        max_question_length=raw_data["max_query_len"],
    )

    return {"features": features, "examples": examples}


def test_BERTQAExtractor(qa_test_features_examples, tmp_path):
    qa_extractor = BERTQAExtractor(cache_dir=tmp_path)

    qa_extractor.fit(features=qa_test_features_examples["features"], cache_model=True, batch_size=8)
    qa_extractor.fit(
        features=qa_test_features_examples["features"],
        cache_model=True,
        warmup_proportion=0.1,
        batch_size=8,
    )

    qa_extractor_from_cache = BERTQAExtractor(cache_dir=tmp_path, load_model_from_dir=tmp_path)

    qa_extractor_from_cache.predict(qa_test_features_examples["features"])

    # Test not overwritting existing model
