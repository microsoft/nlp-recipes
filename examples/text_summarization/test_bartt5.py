import os
import shutil
import sys
from tempfile import TemporaryDirectory
import torch

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.dataset.cnndm import CNNDMBertSumProcessedData, CNNDMSummarizationDataset
from utils_nlp.eval import compute_rouge_python, compute_rouge_perl
from utils_nlp.models.transformers.abstractive_summarization_bartt5 import (
    AbstractiveSummarizer, SummarizationProcessor, validate)

from utils_nlp.models.transformers.datasets import SummarizationDataset
import nltk
from nltk import tokenize

import pandas as pd
import scrapbook as sb
import pprint


QUICK_RUN = True
MODEL_NAME = "bart-large"
CACHE_DIR = "./bartt5_cache" #TemporaryDirectory().name

#processor = SummarizationProcessor(MODEL_NAME,cache_dir=CACHE_DIR ) #tokenizer, config.prefix)
DATA_PATH = "./bartt5_cnndm" #TemporaryDirectory().name
# The number of lines at the head of data file used for preprocessing. -1 means all the lines.
TOP_N = -1
if not QUICK_RUN:
    TOP_N = -1
#abs_sum_train = torch.load(os.path.join(DATA_PATH, "train_full.pt"))
#abs_sum_test = torch.load(os.path.join(DATA_PATH, "test_full.pt"))

BATCH_SIZE_PER_GPU = 1# batch size, unit is the number of samples
MAX_POS_LENGTH = 512
# GPU used for training
NUM_GPUS = torch.cuda.device_count()
# Learning rate
LEARNING_RATE=3e-5
# How often the statistics reports show up in training, unit is step.
REPORT_EVERY=100
SAVE_EVERY=1000
# total number of steps for training
MAX_STEPS=20000
# number of steps for warm up
WARMUP_STEPS=5e2
if not QUICK_RUN:
    MAX_STEPS=2e4
    WARMUP_STEPS=5e2


summarizer = AbstractiveSummarizer(MODEL_NAME, cache_dir=CACHE_DIR)
processor = summarizer.processor
"""
train_dataset, test_dataset = CNNDMSummarizationDataset(top_n=TOP_N, local_cache_path=DATA_PATH, raw=True)
abs_sum_train = processor.preprocess(train_dataset)
abs_sum_test = processor.preprocess(test_dataset)
"""
#torch.save(abs_sum_train,  os.path.join(DATA_PATH, "train_{0}_{1}.pt".format(MODEL_NAME, TOP_N)))
#torch.save(abs_sum_test,  os.path.join(DATA_PATH, "test_{0}_{1}.pt".format(MODEL_NAME, TOP_N)))
abs_sum_train = torch.load(os.path.join(DATA_PATH, "train_{0}_{1}.pt".format(MODEL_NAME, TOP_N)))
abs_sum_test = torch.load(os.path.join(DATA_PATH, "test_{0}_{1}.pt".format(MODEL_NAME, TOP_N)))
def new_validate(summarizer):
    validate(summarizer, abs_sum_test, num_gpus=1)
#"""
summarizer.fit(
            abs_sum_train,
            num_gpus=NUM_GPUS,
            batch_size=BATCH_SIZE_PER_GPU*NUM_GPUS,
            gradient_accumulation_steps=4,
            max_steps=MAX_STEPS,
            max_grad_norm=0.1,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            verbose=True,
            report_every=REPORT_EVERY,
            validation_function=new_validate,
            checkpoint="./bartt5_cache/bart-large_step_10000.pt"
        )
#"""
#prediction = summarizer.predict(abs_sum_test[0:32], num_gpus=NUM_GPUS, batch_size=BATCH_SIZE_PER_GPU*NUM_GPUS)
#print(prediction)
"""summarizer.save_model(global_step=MAX_STEPS,
full_name = os.path.join(
        CACHE_DIR,
        "abssum_{0}_steps_{1}.pt".format(
            MODEL_NAME, MAX_STEPS
        ),
    )
)

"""
