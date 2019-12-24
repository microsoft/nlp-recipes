# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import nltk

nltk.download("punkt")
from nltk import tokenize
import pytest
import os
import shutil


from utils_nlp.models.transformers.datasets import SummarizationDataset
from utils_nlp.models.transformers.extractive_summarization import (
    get_cycled_dataset,
    get_dataloader,
    get_sequential_dataloader,
    ExtractiveSummarizer,
    ExtSumProcessedData,
    ExtSumProcessor,
)

# @pytest.fixture()
def source_data():
    return """Boston, MA welcome to Microsoft/nlp. Welcome to text summarization. Welcome to Microsoft NERD. Look outside, waht a beautiful Charlse River fall view."""


# @pytest.fixture()
def target_data():
    return (
        """ welcome to microsoft/nlp. Welcome to text summarization. Welcome to Microsoft NERD."""
    )

MODEL_NAME = "distilbert-base-uncased"

@pytest.fixture(scope="module")
def data_to_file(tmp_module):
    source = source_data()
    target = target_data()
    source_file = os.path.join(tmp_module, "source.txt")
    target_file = os.path.join(tmp_module, "target.txt")
    f = open(source_file, "w")
    f.write(source)
    f.close()
    f = open(target_file, "w")
    f.write(target)
    f.close()
    train_dataset = SummarizationDataset(
        source_file,
        target_file,
        [tokenize.sent_tokenize],
        [tokenize.sent_tokenize],
        nltk.word_tokenize,
    )
    test_dataset = SummarizationDataset(
        source_file,
        target_file,
        [tokenize.sent_tokenize],
        [tokenize.sent_tokenize],
        nltk.word_tokenize,
    )

    processor = ExtSumProcessor(
        model_name=MODEL_NAME,
        cache_dir=tmp_module,
        max_nsents=200,
        max_src_ntokens=2000,
        min_nsents=0,
        min_src_ntokens=1,
        use_interval=True,
    )
    ext_sum_train = processor.preprocess(
        train_dataset, train_dataset.get_target(), oracle_mode="greedy"
    )
    ext_sum_test = processor.preprocess(
        test_dataset, test_dataset.get_target(), oracle_mode="greedy"
    )

    save_path = os.path.join(tmp_module, "processed")
    train_files = ExtSumProcessedData.save_data(
        ext_sum_train, is_test=False, save_path=save_path, chunk_size=2000
    )
    test_files = ExtSumProcessedData.save_data(
        ext_sum_test, is_test=True, save_path=save_path, chunk_size=2000
    )
    print(train_files)
    print(test_files)
    assert os.path.exists(train_files[0])
    assert os.path.exists(test_files[0])
    return save_path


@pytest.mark.gpu
def test_bert_training(data_to_file, tmp_module):

    CACHE_DIR = tmp_module
    ENCODER = "transformer"
    LEARNING_RATE = 2e-3
    REPORT_EVERY = 100
    MAX_STEPS = 5e2
    WARMUP_STEPS = 1e2
    DATA_SAVED_PATH = data_to_file
    result_base_path = "./results"

    train_dataset_generator, test_dataset_generator = ExtSumProcessedData().splits(
        root=DATA_SAVED_PATH
    )
    summarizer = ExtractiveSummarizer(MODEL_NAME, ENCODER, CACHE_DIR)
    train_dataloader = get_dataloader(
        get_cycled_dataset(train_dataset_generator), is_labeled=True, batch_size=3000
    )
    summarizer.fit(
        train_dataloader,
        num_gpus=1,
        gradient_accumulation_steps=2,
        max_steps=MAX_STEPS,
        lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        verbose=True,
        report_every=REPORT_EVERY,
        clip_grad_norm=False,
    )

    test_dataset = []
    for i in test_dataset_generator():
        test_dataset.extend(i)
    target = [test_dataset[i]["tgt_txt"] for i in range(len(test_dataset))]

    prediction = summarizer.predict(get_sequential_dataloader(test_dataset))
    assert len(prediction) == 1
