# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys
sys.path.insert(0,"../../")


import os
import pytest

from utils_nlp.models.transformers.datasets import SummarizationDataset, SummarizationNonIterableDataset
from utils_nlp.models.transformers.abssum import (
    AbsSumProcessor
)



# @pytest.fixture()
def source_data():
    return (
        "Boston, MA welcome to Microsoft/nlp. Welcome to text summarization."
        "Welcome to Microsoft NERD."
        "Look outside, what a beautiful Charlse River fall view."
    )


# @pytest.fixture()
def target_data():
    return ("welcome to microsoft/nlp." 
            "Welcome to text summarization."
            "Welcome to Microsoft NERD.")


MODEL_NAME = "distilbert-base-uncased"
NUM_GPUS = 1

def test_preprocessing():
    source = source_data()
    target = target_data()
    processor = AbsSumProcessor()
    train_dataset = SummarizationNonIterableDataset([[source]], [[target]])
    batch = processor.collate(train_dataset, 512, "cuda:0")
    print(batch)
    print(len(batch.src[0]))

test_preprocessing()
