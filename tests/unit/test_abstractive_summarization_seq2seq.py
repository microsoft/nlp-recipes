# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pytest

from utils_nlp.models.transformers.abstractive_summarization_seq2seq import (
    S2SAbsSumProcessor, 
    S2SAbstractiveSummarizer, 
    S2SConfig
)

from utils_nlp.models.transformers.datasets import (
    IterableSummarizationDataset,
    SummarizationDataset,
)

MAX_SEQ_LENGTH = 96
MAX_SOURCE_SEQ_LENGTH = 64
MAX_TARGET_SEQ_LENGTH = 16
MAX_TGT_LENGTH = 16

TRAIN_PER_GPU_BATCH_SIZE = 1
TEST_PER_GPU_BATCH_SIZE = 1


@pytest.fixture()
def s2s_test_data():
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


@pytest.mark.gpu
@pytest.mark.parametrize("model_name", ["unilm-base-cased", "minilm-l12-h384-uncased"])
def test_S2SAbstractiveSummarizer(s2s_test_data, tmp, model_name):
    cache_dir = tmp
    model_dir = tmp
    processor = S2SAbsSumProcessor(model_name=model_name, cache_dir=cache_dir)
    train_dataset = processor.s2s_dataset_from_json_or_file(
        s2s_test_data["train_ds"], train_mode=True
    )
    test_dataset = processor.s2s_dataset_from_json_or_file(
        s2s_test_data["test_ds"], train_mode=False
    )
    abs_summarizer = S2SAbstractiveSummarizer(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        max_source_seq_length=MAX_SOURCE_SEQ_LENGTH,
        max_target_seq_length=MAX_TARGET_SEQ_LENGTH,
        cache_dir=cache_dir,
    )

    # test fit and predict
    global_step = abs_summarizer.fit(
        train_dataset,
        per_gpu_batch_size=TRAIN_PER_GPU_BATCH_SIZE,
        save_model_to_dir=model_dir,
    )
    abs_summarizer.predict(
        test_dataset,
        per_gpu_batch_size=TEST_PER_GPU_BATCH_SIZE,
        max_tgt_length=MAX_TGT_LENGTH,
    )

    # test load model from local disk
    abs_summarizer_loaded = S2SAbstractiveSummarizer(
        model_name=model_name,
        load_model_from_dir=model_dir,
        model_file_name="model.{}.bin".format(global_step),
        max_seq_length=MAX_SEQ_LENGTH,
        max_source_seq_length=MAX_SOURCE_SEQ_LENGTH,
        max_target_seq_length=MAX_TARGET_SEQ_LENGTH,
        cache_dir=cache_dir,
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
        save_model_to_dir=model_dir,
        recover_step=global_step,
        recover_dir=model_dir,
        max_steps=global_step + 3,
    )

    abs_summarizer.predict(
        test_dataset,
        per_gpu_batch_size=TEST_PER_GPU_BATCH_SIZE,
        max_tgt_length=MAX_TGT_LENGTH,
    )


def test_S2SAbsSumProcessor(s2s_test_data, tmp):
    expected_output_length = 4
    # prepare files for testing
    train_source_file = os.path.join(tmp, "train.src")
    train_target_file = os.path.join(tmp, "train.tgt")

    test_source_file = os.path.join(tmp, "test.src")

    train_json_file = os.path.join(tmp, "train.json")
    test_json_file = os.path.join(tmp, "test.json")

    with open(train_source_file, "w") as src_file, open(
        train_target_file, "w"
    ) as tgt_file:
        for item in s2s_test_data["train_ds"]:
            src_file.write(item["src"] + "\n")
            tgt_file.write(item["tgt"] + "\n")

    with open(test_source_file, "w") as src_file:
        for item in s2s_test_data["test_ds"]:
            src_file.write(item["src"] + "\n")

    train_iterable_sum_ds = IterableSummarizationDataset(
        source_file=train_source_file, target_file=train_target_file
    )
    test_iterable_sum_ds = IterableSummarizationDataset(source_file=test_source_file)

    train_sum_ds = SummarizationDataset(
        source_file=train_source_file, target_file=train_target_file
    )
    test_sum_ds = SummarizationDataset(source_file=test_source_file)

    train_sum_ds.save_to_jsonl(train_json_file)
    test_sum_ds.save_to_jsonl(test_json_file)

    processor = S2SAbsSumProcessor(cache_dir=tmp)

    train_json_output = processor.s2s_dataset_from_json_or_file(
        input_data=s2s_test_data["train_ds"], train_mode=True
    )
    test_json_output = processor.s2s_dataset_from_json_or_file(
        input_data=s2s_test_data["test_ds"], train_mode=False
    )

    assert len(train_json_output) == expected_output_length
    assert len(test_json_output) == expected_output_length

    train_file_output = processor.s2s_dataset_from_json_or_file(
        input_data=train_json_file, train_mode=True
    )
    test_file_output = processor.s2s_dataset_from_json_or_file(
        input_data=test_json_file, train_mode=False
    )

    assert len(train_file_output) == expected_output_length
    assert len(test_file_output) == expected_output_length

    train_iterable_sum_ds_output = processor.s2s_dataset_from_iterable_sum_ds(
        sum_ds=train_iterable_sum_ds, train_mode=True
    )
    test_iterable_sum_ds_output = processor.s2s_dataset_from_iterable_sum_ds(
        sum_ds=test_iterable_sum_ds, train_mode=False
    )

    assert len(train_iterable_sum_ds_output) == expected_output_length
    assert len(test_iterable_sum_ds_output) == expected_output_length

    train_sum_ds_output = processor.s2s_dataset_from_sum_ds(
        sum_ds=train_sum_ds, train_mode=True
    )
    test_sum_ds_output = processor.s2s_dataset_from_sum_ds(
        sum_ds=test_sum_ds, train_mode=False
    )

    assert len(train_sum_ds_output) == expected_output_length
    assert len(test_sum_ds_output) == expected_output_length


def test_S2SConfig(tmp):
    config_file = os.path.join(tmp, "s2s_config.json")

    config = S2SConfig()

    config.save_to_json(config_file)

    loaded_config = S2SConfig.load_from_json(config_file)

    assert loaded_config.__dict__ == config.__dict__
