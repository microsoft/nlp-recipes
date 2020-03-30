# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import nltk
import pytest
from nltk import tokenize

from utils_nlp.models.transformers.datasets import SummarizationDataset
from utils_nlp.models.transformers.extractive_summarization import (
    ExtractiveSummarizer,
    ExtSumProcessor,
)

nltk.download("punkt")


# @pytest.fixture()
def source_data():
    return (
        "Boston, MA welcome to Microsoft/nlp. Welcome to text summarization."
        "Welcome to Microsoft NERD."
        "Look outside, what a beautiful Charlse River fall view."
    )


# @pytest.fixture()
def target_data():
    return (
        "welcome to microsoft/nlp."
        "Welcome to text summarization."
        "Welcome to Microsoft NERD."
    )


MODEL_NAME = "distilbert-base-uncased"

@pytest.fixture(scope="module")
def data(tmp_module):
    source = source_data()
    target = target_data()
    train_dataset = SummarizationDataset(
        None,
        source=[source],
        target=[target],
        source_preprocessing=[tokenize.sent_tokenize],
        target_preprocessing=[tokenize.sent_tokenize],
        word_tokenize=nltk.word_tokenize,
    )
    test_dataset = SummarizationDataset(
        None,
        source=[source],
        source_preprocessing=[tokenize.sent_tokenize],
        word_tokenize=nltk.word_tokenize,
    )

    processor = ExtSumProcessor(
        model_name=MODEL_NAME,
        cache_dir=tmp_module,
        max_nsents=200,
        max_src_ntokens=2000,
        min_nsents=0,
        min_src_ntokens=1,
    )
    ext_sum_train = processor.preprocess(train_dataset, oracle_mode="greedy")
    ext_sum_test = processor.preprocess(test_dataset, oracle_mode="greedy")
    return processor, ext_sum_train, ext_sum_test


@pytest.mark.gpu
def test_bert_training(data, tmp_module):

    CACHE_DIR = tmp_module
    ENCODER = "transformer"
    MAX_POS = 768
    BATCH_SIZE = 128
    LEARNING_RATE = 2e-3
    REPORT_EVERY = 50
    MAX_STEPS = 20
    WARMUP_STEPS = 1e2

    processor, train_dataset, test_dataset = data
    summarizer = ExtractiveSummarizer(
        processor, MODEL_NAME, ENCODER, MAX_POS, CACHE_DIR
    )
    summarizer.fit(
        train_dataset,
        num_gpus=None,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        max_steps=MAX_STEPS,
        lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        verbose=True,
        report_every=REPORT_EVERY,
        clip_grad_norm=False,
    )

    prediction = summarizer.predict(test_dataset, num_gpus=None, batch_size=128)
    assert len(prediction) == 1
