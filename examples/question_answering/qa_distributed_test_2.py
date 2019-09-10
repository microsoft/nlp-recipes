QUICK_RUN = False

import os
import sys

import torch
import numpy as np

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.dataset.squad import load_pandas_df
from utils_nlp.models.bert.common import Language
from utils_nlp.models.bert.question_answering_distributed_v1 import BERTQAExtractor
from utils_nlp.models.bert.qa_utils import postprocess_answer, evaluate_qa
from utils_nlp.common.timer import Timer

TRAIN_DATA_USED_PERCENT = 1
DEV_DATA_USED_PERCENT = 1
NUM_EPOCHS = 2

if QUICK_RUN:
    TRAIN_DATA_USED_PERCENT = 0.001
    DEV_DATA_USED_PERCENT = 0.01
    NUM_EPOCHS = 1

if torch.cuda.is_available() and torch.cuda.device_count() >= 4:
    MAX_SEQ_LENGTH = 384
    DOC_STRIDE = 128
    BATCH_SIZE = 4
else:
    MAX_SEQ_LENGTH = 128
    DOC_STRIDE = 64
    BATCH_SIZE = 4

print("Max sequence length: {}".format(MAX_SEQ_LENGTH))
print("Document stride: {}".format(DOC_STRIDE))
print("Batch size: {}".format(BATCH_SIZE))

SQUAD_VERSION = "v1.1"
CACHE_DIR = "./temp"

LANGUAGE = Language.ENGLISHLARGEWWM
DO_LOWER_CASE = True

MAX_QUESTION_LENGTH = 64
LEARNING_RATE = 3e-5

DOC_TEXT_COL = "doc_text"
QUESTION_TEXT_COL = "question_text"
ANSWER_START_COL = "answer_start"
ANSWER_TEXT_COL = "answer_text"
QA_ID_COL = "qa_id"
IS_IMPOSSIBLE_COL = "is_impossible"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(RANDOM_SEED)


train_df = load_pandas_df(local_cache_path=".", squad_version="v1.1", file_split="train")
dev_df = load_pandas_df(local_cache_path=".", squad_version="v1.1", file_split="dev")

train_features = torch.load("./temp/cached_features_train")
qa_examples = torch.load("temp/cached_examples_train")

dev_features = torch.load("./temp/cached_features")
dev_examples = torch.load("./temp/cached_examples")

qa_extractor = BERTQAExtractor(language=LANGUAGE, cache_dir=CACHE_DIR)

with Timer() as t:
    qa_extractor.fit(
        train_features,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=16,
        cache_model=True,
        distributed=True,
    )
print("Training time : {:.3f} hrs".format(t.interval / 3600))

qa_results = qa_extractor.predict(dev_features, batch_size=BATCH_SIZE)

final_answers, answer_probs, nbest_answers = postprocess_answer(
    qa_results, dev_examples, dev_features, do_lower_case=DO_LOWER_CASE
)

evaluation_result = evaluate_qa(
    qa_ids=dev_df["qa_id"], actuals=dev_df["answer_text"], preds=final_answers
)
