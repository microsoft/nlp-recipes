# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torch

from utils_nlp.common.pytorch_utils import dataloader_from_dataset
from utils_nlp.models.transformers.datasets import QADataset
from utils_nlp.models.transformers.question_answering import (
    CACHED_EXAMPLES_TEST_FILE,
    CACHED_FEATURES_TEST_FILE,
    AnswerExtractor,
    QAProcessor,
)

NUM_GPUS = max(1, torch.cuda.device_count())
BATCH_SIZE = 8


@pytest.fixture(scope="module")
def qa_test_data(qa_test_df, tmp_module):

    train_dataset = QADataset(
        df=qa_test_df["test_df"],
        doc_text_col=qa_test_df["doc_text_col"],
        question_text_col=qa_test_df["question_text_col"],
        answer_start_col=qa_test_df["answer_start_col"],
        answer_text_col=qa_test_df["answer_text_col"],
        qa_id_col=qa_test_df["qa_id_col"],
    )

    train_dataset_list = QADataset(
        df=qa_test_df["test_df"],
        doc_text_col=qa_test_df["doc_text_col"],
        question_text_col=qa_test_df["question_text_col"],
        answer_start_col=qa_test_df["answer_start_list_col"],
        answer_text_col=qa_test_df["answer_text_list_col"],
        qa_id_col=qa_test_df["qa_id_col"],
    )

    train_dataset_start_text_mismatch = QADataset(
        df=qa_test_df["test_df"],
        doc_text_col=qa_test_df["doc_text_col"],
        question_text_col=qa_test_df["question_text_col"],
        answer_start_col=qa_test_df["answer_start_list_col"],
        answer_text_col=qa_test_df["answer_text_col"],
        qa_id_col=qa_test_df["qa_id_col"],
    )

    train_dataset_multi_answers = QADataset(
        df=qa_test_df["test_df"],
        doc_text_col=qa_test_df["doc_text_col"],
        question_text_col=qa_test_df["question_text_col"],
        answer_start_col=qa_test_df["answer_start_multi_col"],
        answer_text_col=qa_test_df["answer_text_multi_col"],
        qa_id_col=qa_test_df["qa_id_col"],
    )

    test_dataset = QADataset(
        df=qa_test_df["test_df"],
        doc_text_col=qa_test_df["doc_text_col"],
        question_text_col=qa_test_df["question_text_col"],
        qa_id_col=qa_test_df["qa_id_col"],
    )

    # bert
    qa_processor_bert = QAProcessor(cache_dir=tmp_module)
    train_features_bert = qa_processor_bert.preprocess(
        train_dataset,
        is_training=True,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp_module,
    )
    test_features_bert = qa_processor_bert.preprocess(
        test_dataset,
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp_module,
    )

    # xlnet
    qa_processor_xlnet = QAProcessor(
        model_name="xlnet-base-cased", cache_dir=tmp_module
    )
    train_features_xlnet = qa_processor_xlnet.preprocess(
        train_dataset,
        is_training=True,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp_module,
    )
    test_features_xlnet = qa_processor_xlnet.preprocess(
        test_dataset,
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp_module,
    )

    # distilbert
    qa_processor_distilbert = QAProcessor(
        model_name="distilbert-base-uncased", cache_dir=tmp_module
    )
    train_features_distilbert = qa_processor_distilbert.preprocess(
        train_dataset,
        is_training=True,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp_module,
    )
    test_features_distilbert = qa_processor_distilbert.preprocess(
        test_dataset,
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp_module,
    )

    return {
        "train_dataset": train_dataset,
        "train_dataset_list": train_dataset_list,
        "train_dataset_start_text_mismatch": train_dataset_start_text_mismatch,
        "train_dataset_multi_answers": train_dataset_multi_answers,
        "test_dataset": test_dataset,
        "train_features_bert": train_features_bert,
        "test_features_bert": test_features_bert,
        "train_features_xlnet": train_features_xlnet,
        "test_features_xlnet": test_features_xlnet,
        "train_features_distilbert": train_features_distilbert,
        "test_features_distilbert": test_features_distilbert,
    }


@pytest.mark.gpu
def test_QAProcessor(qa_test_data, tmp_module):
    for model_name in [
        "bert-base-cased",
        "xlnet-base-cased",
        "distilbert-base-uncased",
    ]:
        qa_processor = QAProcessor(model_name=model_name, cache_dir=tmp_module)
        qa_processor.preprocess(
            qa_test_data["train_dataset"],
            is_training=True,
            feature_cache_dir=tmp_module,
        )
        qa_processor.preprocess(
            qa_test_data["train_dataset_list"],
            is_training=True,
            feature_cache_dir=tmp_module,
        )
        qa_processor.preprocess(
            qa_test_data["test_dataset"],
            is_training=False,
            feature_cache_dir=tmp_module,
        )

    # test unsupported model type
    with pytest.raises(ValueError):
        qa_processor = QAProcessor(model_name="abc", cache_dir=tmp_module)

    # test training data has no ground truth exception
    with pytest.raises(Exception):
        qa_processor.preprocess(
            qa_test_data["test_dataset"], is_training=True, feature_cache_dir=tmp_module
        )

    # test when answer start is a list, but answer text is not
    with pytest.raises(Exception):
        qa_processor.preprocess(
            qa_test_data["train_dataset_start_text_mismatch"],
            is_training=True,
            feature_cache_dir=tmp_module,
        )

    # test when training data has multiple answers
    with pytest.raises(Exception):
        qa_processor.preprocess(
            qa_test_data["train_dataset_multi_answers"],
            is_training=True,
            feature_cache_dir=tmp_module,
        )


def test_AnswerExtractor(qa_test_data, tmp_module):
    # bert
    qa_extractor_bert = AnswerExtractor(cache_dir=tmp_module)
    train_loader_bert = dataloader_from_dataset(qa_test_data["train_features_bert"])
    test_loader_bert = dataloader_from_dataset(
        qa_test_data["test_features_bert"], shuffle=False
    )
    qa_extractor_bert.fit(train_loader_bert, verbose=False, cache_model=True)

    # test saving fine-tuned model
    model_output_dir = os.path.join(tmp_module, "fine_tuned")
    assert os.path.exists(os.path.join(model_output_dir, "pytorch_model.bin"))
    assert os.path.exists(os.path.join(model_output_dir, "config.json"))

    qa_extractor_from_cache = AnswerExtractor(
        cache_dir=tmp_module, load_model_from_dir=model_output_dir
    )
    qa_extractor_from_cache.predict(test_loader_bert, verbose=False)

    # xlnet
    train_loader_xlnet = dataloader_from_dataset(qa_test_data["train_features_xlnet"])
    test_loader_xlnet = dataloader_from_dataset(
        qa_test_data["test_features_xlnet"], shuffle=False
    )
    qa_extractor_xlnet = AnswerExtractor(
        model_name="xlnet-base-cased", cache_dir=tmp_module
    )
    qa_extractor_xlnet.fit(train_loader_xlnet, verbose=False, cache_model=False)
    qa_extractor_xlnet.predict(test_loader_xlnet, verbose=False)

    # distilbert
    train_loader_xlnet = dataloader_from_dataset(
        qa_test_data["train_features_distilbert"]
    )
    test_loader_xlnet = dataloader_from_dataset(
        qa_test_data["test_features_distilbert"], shuffle=False
    )
    qa_extractor_distilbert = AnswerExtractor(
        model_name="distilbert-base-uncased", cache_dir=tmp_module
    )
    qa_extractor_distilbert.fit(train_loader_xlnet, verbose=False, cache_model=False)
    qa_extractor_distilbert.predict(test_loader_xlnet, verbose=False)


def test_postprocess_bert_answer(qa_test_data, tmp_module):
    qa_processor = QAProcessor(cache_dir=tmp_module)
    test_features = qa_processor.preprocess(
        qa_test_data["test_dataset"],
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp_module,
    )
    test_loader = dataloader_from_dataset(test_features, shuffle=False)
    qa_extractor = AnswerExtractor(cache_dir=tmp_module)
    predictions = qa_extractor.predict(test_loader)

    qa_processor.postprocess(
        results=predictions,
        examples_file=os.path.join(tmp_module, CACHED_EXAMPLES_TEST_FILE),
        features_file=os.path.join(tmp_module, CACHED_FEATURES_TEST_FILE),
        output_prediction_file=os.path.join(tmp_module, "qa_predictions.json"),
        output_nbest_file=os.path.join(tmp_module, "nbest_predictions.json"),
        output_null_log_odds_file=os.path.join(tmp_module, "null_odds.json"),
    )

    qa_processor.postprocess(
        results=predictions,
        examples_file=os.path.join(tmp_module, CACHED_EXAMPLES_TEST_FILE),
        features_file=os.path.join(tmp_module, CACHED_FEATURES_TEST_FILE),
        unanswerable_exists=True,
        verbose_logging=True,
        output_prediction_file=os.path.join(tmp_module, "qa_predictions.json"),
        output_nbest_file=os.path.join(tmp_module, "nbest_predictions.json"),
        output_null_log_odds_file=os.path.join(tmp_module, "null_odds.json"),
    )


def test_postprocess_xlnet_answer(qa_test_data, tmp_module):
    qa_processor = QAProcessor(model_name="xlnet-base-cased", cache_dir=tmp_module)
    test_features = qa_processor.preprocess(
        qa_test_data["test_dataset"],
        is_training=False,
        max_question_length=16,
        max_seq_length=64,
        doc_stride=32,
        feature_cache_dir=tmp_module,
    )
    test_loader = dataloader_from_dataset(test_features, shuffle=False)
    qa_extractor = AnswerExtractor(model_name="xlnet-base-cased", cache_dir=tmp_module)
    predictions = qa_extractor.predict(test_loader)

    qa_processor.postprocess(
        results=predictions,
        examples_file=os.path.join(tmp_module, CACHED_EXAMPLES_TEST_FILE),
        features_file=os.path.join(tmp_module, CACHED_FEATURES_TEST_FILE),
        output_prediction_file=os.path.join(tmp_module, "qa_predictions.json"),
        output_nbest_file=os.path.join(tmp_module, "nbest_predictions.json"),
        output_null_log_odds_file=os.path.join(tmp_module, "null_odds.json"),
    )

    qa_processor.postprocess(
        results=predictions,
        examples_file=os.path.join(tmp_module, CACHED_EXAMPLES_TEST_FILE),
        features_file=os.path.join(tmp_module, CACHED_FEATURES_TEST_FILE),
        unanswerable_exists=True,
        verbose_logging=True,
        output_prediction_file=os.path.join(tmp_module, "qa_predictions.json"),
        output_nbest_file=os.path.join(tmp_module, "nbest_predictions.json"),
        output_null_log_odds_file=os.path.join(tmp_module, "null_odds.json"),
    )
