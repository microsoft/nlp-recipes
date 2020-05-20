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
from utils_nlp.models.transformers.abstractive_summarization_bartt5 import (
    AbstractiveSummarizer)

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
def test_dataset_for_bartt5(tmp_module):
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
    )
    test_dataset = SummarizationDataset(
        source_file = source_file,
        target_file = target_file,
    )
    return train_dataset, test_dataset


@pytest.mark.gpu
@pytest.fixture()
def test_train_model(tmp_module, test_dataset_for_bartt5, batch_size=1):
    CACHE_PATH = (
        tmp_module  
    )
    DATA_PATH = (
        tmp_module 
    )
    MODEL_PATH = (
        tmp_module
    )

    summarizer = AbstractiveSummarizer("t5-small", cache_dir=CACHE_PATH)

    checkpoint = None
    train_sum_dataset, _ = test_dataset_for_bartt5
    abs_sum_train = summarizer.processor.preprocess(train_sum_dataset)

    MAX_STEP = 20
    TOP_N = 8
    summarizer.fit(
        abs_sum_train,
        batch_size=batch_size,
        max_steps=MAX_STEP,
        local_rank=-1,
        learning_rate=0.002,
        warmup_steps=20000,
        num_gpus=None,
        report_every=10,
        save_every=100,
        fp16=False,
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
    test_dataset_for_bartt5,
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

    _, test_sum_dataset = test_dataset_for_bartt5

    summarizer = AbstractiveSummarizer("t5-small", cache_dir=CACHE_PATH)
    abs_sum_test = summarizer.processor.preprocess(test_sum_dataset)
    checkpoint = torch.load(test_train_model, map_location="cpu")

    summarizer.model.load_state_dict(checkpoint["model"])
    
    reference_summaries = [
        "".join(i["tgt"]).rstrip("\n") for i in  abs_sum_test
    ]
    print("start prediction")
    generated_summaries = summarizer.predict(
        abs_sum_test, batch_size=batch_size, num_gpus=None
    )

    def _write_list_to_file(list_items, filename):
        with open(filename, "w") as filehandle:
            # for cnt, line in enumerate(filehandle):
            for item in list_items:
                filehandle.write("%s\n" % item)

    print("writing generated summaries")
    _write_list_to_file(generated_summaries, os.path.join(CACHE_PATH, "prediction.txt"))

    assert len(generated_summaries) == len(reference_summaries)
