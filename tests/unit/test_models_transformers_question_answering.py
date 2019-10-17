# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
from utils_nlp.dataset.pytorch import QADataset
from utils_nlp.models.transformers.question_answering import (
    QAProcessor,
    AnswerExtractor,
    postprocess_xlnet_answer,
    postprocess_bert_answer,
    CACHED_EXAMPLES_TEST_FILE,
    CACHED_FEATURES_TEST_FILE,
)


@pytest.fixture()
def qa_test_data(qa_test_df, tmp):

    train_dataset = QADataset(
        df=qa_test_df["test_df"],
        doc_text_col=qa_test_df["doc_text_col"],
        question_text_col=qa_test_df["question_text_col"],
        answer_start_col=qa_test_df["answer_start_col"],
        answer_text_col=qa_test_df["answer_text_col"],
        qa_id_col=qa_test_df["qa_id_col"],
    )

    test_dataset = QADataset(
        df=qa_test_df["test_df"],
        doc_text_col=qa_test_df["doc_text_col"],
        question_text_col=qa_test_df["question_text_col"],
        qa_id_col=qa_test_df["qa_id_col"],
    )

    qa_processor_bert = QAProcessor()
    train_features_bert = qa_processor_bert.preprocess(
        train_dataset,
        is_training=True,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp,
    )

    test_features_bert = qa_processor_bert.preprocess(
        test_dataset,
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp,
    )

    qa_processor_xlnet = QAProcessor(model_name="xlnet-base-cased")
    train_features_xlnet = qa_processor_xlnet.preprocess(
        train_dataset,
        is_training=True,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp,
    )

    test_features_xlnet = qa_processor_xlnet.preprocess(
        test_dataset,
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp,
    )

    qa_processor_distilbert = QAProcessor(model_name="distilbert-base-uncased")
    train_features_distilbert = qa_processor_distilbert.preprocess(
        train_dataset,
        is_training=True,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp,
    )

    test_features_distilbert = qa_processor_distilbert.preprocess(
        test_dataset,
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp,
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_features_bert": train_features_bert,
        "test_features_bert": test_features_bert,
        "train_features_xlnet": train_features_xlnet,
        "test_features_xlnet": test_features_xlnet,
        "train_features_distilbert": train_features_distilbert,
        "test_features_distilbert": test_features_distilbert,
    }


def test_QAProcessor(qa_test_data, tmp):
    for model_name in ["bert-base-cased", "xlnet-base-cased", "distilbert-base-uncased"]:
        qa_processor = QAProcessor(model_name=model_name)
        qa_processor.preprocess(qa_test_data["train_dataset"], is_training=True)
        qa_processor.preprocess(qa_test_data["test_dataset"], is_training=False)

    # test unsupproted model type
    with pytest.raises(ValueError):
        qa_processor = QAProcessor(model_name="abc")

    # test training data has no ground truth exception
    with pytest.raises(Exception):
        qa_processor.preprocess(qa_test_data["test_dataset"], is_training=True)


def test_AnswerExtractor(qa_test_data, tmp):
    # test bert
    qa_extractor_bert = AnswerExtractor(cache_dir=tmp)
    qa_extractor_bert.fit(
        qa_test_data["train_features_bert"], cache_model=True, per_gpu_batch_size=8
    )

    # test saving fine-tuned model
    model_output_dir = os.path.join(tmp, "fine_tuned")
    assert os.path.exists(os.path.join(model_output_dir, "pytorch_model.bin"))
    assert os.path.exists(os.path.join(model_output_dir, "config.json"))

    qa_extractor_from_cache = AnswerExtractor(
        cache_dir=tmp, load_model_from_dir=model_output_dir
    )
    qa_extractor_from_cache.predict(qa_test_data["test_features_bert"])

    qa_extractor_xlnet = AnswerExtractor(model_name="xlnet-base-cased", cache_dir=tmp)
    qa_extractor_xlnet.fit(
        qa_test_data["train_features_xlnet"], cache_model=False, per_gpu_batch_size=8
    )
    qa_extractor_xlnet.predict(qa_test_data["test_features_xlnet"])

    qa_extractor_distilbert = AnswerExtractor(
        model_name="distilbert-base-uncased", cache_dir=tmp
    )
    qa_extractor_distilbert.fit(
        qa_test_data["train_features_distilbert"], cache_model=False, per_gpu_batch_size=8
    )
    qa_extractor_distilbert.predict(qa_test_data["test_features_distilbert"])


def test_postprocess_bert_answer(qa_test_data, tmp):
    qa_processor = QAProcessor()
    test_features = qa_processor.preprocess(
        qa_test_data["test_dataset"],
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp,
    )
    qa_extractor = AnswerExtractor(cache_dir=tmp)
    predictions = qa_extractor.predict(test_features)

    qa_processor.postprocess(
        results=predictions,
        examples_file=os.path.join(tmp, CACHED_EXAMPLES_TEST_FILE),
        features_file=os.path.join(tmp, CACHED_FEATURES_TEST_FILE),
    )

    qa_processor.postprocess(
        results=predictions,
        examples_file=os.path.join(tmp, CACHED_EXAMPLES_TEST_FILE),
        features_file=os.path.join(tmp, CACHED_FEATURES_TEST_FILE),
        unanswerable_exists=True,
        verbose_logging=True,
    )


def test_postprocess_xlnet_answer(qa_test_data, tmp):
    qa_processor = QAProcessor(model_name="xlnet-base-cased")
    test_features = qa_processor.preprocess(
        qa_test_data["test_dataset"],
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp,
    )
    qa_extractor = AnswerExtractor(model_name="xlnet-base-cased", cache_dir=tmp)
    predictions = qa_extractor.predict(test_features)

    qa_processor.postprocess(
        results=predictions,
        examples_file=os.path.join(tmp, CACHED_EXAMPLES_TEST_FILE),
        features_file=os.path.join(tmp, CACHED_FEATURES_TEST_FILE),
    )

    qa_processor.postprocess(
        results=predictions,
        examples_file=os.path.join(tmp, CACHED_EXAMPLES_TEST_FILE),
        features_file=os.path.join(tmp, CACHED_FEATURES_TEST_FILE),
        unanswerable_exists=True,
        verbose_logging=True,
    )
