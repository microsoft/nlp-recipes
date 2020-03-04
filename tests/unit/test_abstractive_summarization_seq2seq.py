# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from utils_nlp.models import (
    S2SAbsSumProcessor,
    S2SAbstractiveSummarizer,
)

MAX_SEQ_LENGTH = 96
MAX_SOURCE_SEQ_LENGTH = 64
MAX_TARGET_SEQ_LENGTH = 16
MAX_TGT_LENGTH = 16

TRAIN_PER_GPU_BATCH_SIZE = 1
TEST_PER_GPU_BATCH_SIZE = 1


@pytest.fixture()
def s2s_test_data(tmp):
    train_ds = [
        {
            "src": "Moscow is usually blanketed in snow for four to five months "
            "a year. But this year, Russia's capital had barely any snow cover "
            "in the whole of February.",
            "tgt": "Mowcow is unusually snowless this February.",
        },
        {
            "src": "US stocks rallied back to life on Wednesday, retracing "
            "losses from the previous day over coronavirus fears.",
            "tgt": "US stocks retraced losses on Wednesday.",
        },
        {
            "src": "The Los Angeles County Board of Supervisors and the "
            "Department of Public Health have declared a local and public "
            "health emergency in response to the spread of coronavirus across "
            "the country, which includes six additional cases in L.A. County.",
            "tgt": "Los Angeles County declares health emergency due to "
            "coronavirus concerns.",
        },
        {
            "src": "Tree cover in US cities is shrinking. A study published last "
            "year by the US Forest Service found that we lost 36 million trees "
            "annually from urban and rural communities over a five-year period. "
            "That's a 1% drop from 2009 to 2014",
            "tgt": "US cities are losing 36 million trees a year.",
        },
    ]
    test_ds = [
        {
            "src": "A 5-year-old student at an elementary school in Vista, "
            "California, collected enough money to pay off the negative lunch "
            "balances of 123 students at her school."
        },
        {
            "src": "As counting gets underway in Israel's unprecedented third "
            "election in 11 months, initial exit polls projected Prime Minister "
            "Benjamin Netanyahu's Likud party as the winners."
        },
        {
            "src": "The German automaker's refreshed logo ditches the black ring "
            "for a transparent circle. The rest of it, including the typeface, "
            "has a flatter and more modern look. The blue and white emblem inside "
            "the ring remains."
        },
        {
            "src": "Before dawn Tuesday, 24 people were killed, and hundreds of "
            "buildings were destroyed by the storms. Officials in Putnam County, "
            "which suffered 18 storm-related deaths, said they are working to "
            "locate 17 people who are unaccounted for, down from 38 earlier in the day."
        },
    ]

    return {"train_ds": train_ds, "test_ds": test_ds}


def test_S2SAbstractiveSummarizer(s2s_test_data, tmp):
    processor = S2SAbsSumProcessor(cache_dir=tmp)
    train_dataset = processor.s2s_dataset_from_sum_ds(
        s2s_test_data["train_ds"], train_mode=True
    )
    test_dataset = processor.s2s_dataset_from_sum_ds(
        s2s_test_data["test_ds"], train_mode=False
    )
    abs_summarizer = S2SAbstractiveSummarizer(
        max_seq_length=MAX_SEQ_LENGTH,
        max_source_seq_length=MAX_SOURCE_SEQ_LENGTH,
        max_target_seq_length=MAX_TARGET_SEQ_LENGTH,
        cache_dir=tmp,
    )

    # test fit and predict
    abs_summarizer.fit(
        train_dataset,
        per_gpu_batch_size=TRAIN_PER_GPU_BATCH_SIZE,
        save_model_to_dir=tmp,
    )
    abs_summarizer.predict(
        test_dataset,
        per_gpu_batch_size=TEST_PER_GPU_BATCH_SIZE,
        max_tgt_length=MAX_TGT_LENGTH,
    )

    # test load model from local disk
    abs_summarizer_loaded = S2SAbstractiveSummarizer(
        load_model_from_dir=tmp,
        model_file_name="model.1.bin",
        max_seq_length=MAX_SEQ_LENGTH,
        max_source_seq_length=MAX_SOURCE_SEQ_LENGTH,
        max_target_seq_length=MAX_TARGET_SEQ_LENGTH,
        cache_dir=tmp,
    )

    abs_summarizer_loaded.predict(
        test_dataset,
        per_gpu_batch_size=TEST_PER_GPU_BATCH_SIZE,
        max_tgt_length=MAX_TGT_LENGTH,
    )

    # test recover model
    abs_summarizer.fit(
        train_dataset,
        per_gpu_batch_size=TRAIN_PER_GPU_BATCH_SIZE,
        save_model_to_dir=tmp,
        recover_step=1,
        recover_dir=tmp,
        max_steps=4,
    )

    abs_summarizer.predict(
        test_dataset,
        per_gpu_batch_size=TEST_PER_GPU_BATCH_SIZE,
        max_tgt_length=MAX_TGT_LENGTH,
    )
