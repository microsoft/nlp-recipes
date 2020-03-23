# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import nltk
from nltk import tokenize
import os
import pytest
import torch

torch.set_printoptions(threshold=5000)

from utils_nlp.models.transformers.datasets import SummarizationDataset
from utils_nlp.models.transformers.abstractive_summarization_bertsum import (
    BertSumAbs,
    BertSumAbsProcessor,
    validate,
)

# @pytest.fixture()
def source_data():
    return [
        [
            "Boston, MA welcome to Microsoft/nlp",
            "Welcome to text summarization.",
            "Welcome to Microsoft NERD.",
            "Look outside, what a beautiful Charlse River fall view.",
        ],
        ["I am just another test case"],
        ["want to test more"],
    ]


# @pytest.fixture()
def target_data():
    return [
        [
            "welcome to microsoft/nlp.",
            "Welcome to text summarization.",
            "Welcome to Microsoft NERD.",
        ],
        ["I am just another test summary"],
        ["yest, I agree"],
    ]


#NUM_GPUS = 2
os.environ["NCCL_IB_DISABLE"] = "0"


@pytest.fixture(scope="module")
def test_dataset_for_bertsumabs(tmp_module):
    source = source_data()
    target = target_data()
    source_file = os.path.join(tmp_module, "source.txt")
    target_file = os.path.join(tmp_module, "target.txt")
    f = open(source_file, "w")
    for i in source:
        f.write(" ".join(i))
        f.write("\n")
    f.close()
    f = open(target_file, "w")
    for i in target:
        f.write(" ".join(i))
        f.write("\n")
    f.close()
    train_dataset = SummarizationDataset(
        source_file = source_file,
        target_file = target_file,
        source_preprocessing = [tokenize.sent_tokenize],
        target_preprocessing = [tokenize.sent_tokenize],
    )
    test_dataset = SummarizationDataset(
        source_file = source_file,
        target_file = target_file,
        source_preprocessing = [tokenize.sent_tokenize],
        target_preprocessing = [tokenize.sent_tokenize],
    )
    processor = BertSumAbsProcessor(cache_dir=tmp_module)
    batch = processor.collate(train_dataset, 512, "cuda:0")
    assert len(batch.src) == 3
    return train_dataset, test_dataset


@pytest.mark.gpu
@pytest.fixture()
def test_train_model(tmp_module, test_dataset_for_bertsumabs, batch_size=1):
    CACHE_PATH = (
        tmp_module  
    )
    DATA_PATH = (
        tmp_module 
    )
    MODEL_PATH = (
        tmp_module
    )

    processor = BertSumAbsProcessor(cache_dir=CACHE_PATH)
    summarizer = BertSumAbs(processor, cache_dir=CACHE_PATH)

    checkpoint = None
    train_sum_dataset, test_sum_dataset = test_dataset_for_bertsumabs

    def this_validate(class_obj):
        return validate(class_obj, test_sum_dataset)

    MAX_STEP = 20
    TOP_N = 8
    summarizer.fit(
        train_sum_dataset,
        batch_size=batch_size,
        max_steps=MAX_STEP,
        local_rank=-1,
        learning_rate_bert=0.002,
        learning_rate_dec=0.2,
        warmup_steps_bert=20000,
        warmup_steps_dec=10000,
        num_gpus=None,
        report_every=10,
        save_every=100,
        validation_function=this_validate,
        fp16=False,
        fp16_opt_level="O1",
        checkpoint=checkpoint,
    )
    saved_model_path = os.path.join(
        MODEL_PATH, "summarizer_step_{}.pt".format(MAX_STEP)
    )
    summarizer.save_model(MAX_STEP, saved_model_path)

    return saved_model_path


@pytest.mark.gpu
def test_finetuned_model(
    tmp_module,
    test_train_model,
    test_dataset_for_bertsumabs,
    top_n=8,
    batch_size=1,
):
    CACHE_PATH = (
        tmp_module  
    )
    DATA_PATH = (
        tmp_module 
    )
    MODEL_PATH = (
        tmp_module
    )

    # train_sum_dataset, test_sum_dataset = preprocess_cnndm_abs(need_process=False)
    train_sum_dataset, test_sum_dataset = test_dataset_for_bertsumabs

    processor = BertSumAbsProcessor(cache_dir=CACHE_PATH)
    checkpoint = torch.load(test_train_model)

    summarizer = BertSumAbs(
        processor,
        cache_dir=CACHE_PATH,
        test=True,
        max_pos_length=checkpoint["max_pos_length"],
    )
    summarizer.model.load_checkpoint(checkpoint["model"])
    
    shortened_dataset = test_sum_dataset.shorten(top_n)
    reference_summaries = [
        "".join(t).rstrip("\n") for t in shortened_dataset.get_target()
    ]
    print("start prediction")
    generated_summaries = summarizer.predict(
        shortened_dataset, batch_size=batch_size, num_gpus=None
    )

    def _write_list_to_file(list_items, filename):
        with open(filename, "w") as filehandle:
            # for cnt, line in enumerate(filehandle):
            for item in list_items:
                filehandle.write("%s\n" % item)

    print("writing generated summaries")
    _write_list_to_file(generated_summaries, os.path.join(CACHE_PATH, "prediction.txt"))

    assert len(generated_summaries) == len(reference_summaries)
