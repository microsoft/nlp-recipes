QUICK_RUN = False

import os
import sys

import torch
import numpy as np

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.dataset.squad import load_pandas_df
from utils_nlp.models.bert.common import Language, Tokenizer
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
    BATCH_SIZE = 16
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

train_df = train_df.sample(frac=TRAIN_DATA_USED_PERCENT).reset_index(drop=True)
dev_df = dev_df.sample(frac=DEV_DATA_USED_PERCENT).reset_index(drop=True)

tokenizer = Tokenizer(language=LANGUAGE, to_lower=DO_LOWER_CASE, cache_dir=CACHE_DIR)

qa_extractor=BERTQAExtractor(language=LANGUAGE, cache_dir=CACHE_DIR, load_model_from_dir="./temp/distributed_0")

dev_features, dev_examples = tokenizer.tokenize_qa(
    doc_text=dev_df[DOC_TEXT_COL], 
    question_text=dev_df[QUESTION_TEXT_COL], 
    answer_start=dev_df[ANSWER_START_COL], 
    answer_text=dev_df[ANSWER_TEXT_COL],
    qa_id=dev_df[QA_ID_COL],
    is_impossible=dev_df[IS_IMPOSSIBLE_COL],
    is_training=False,
    max_len=MAX_SEQ_LENGTH,
    max_question_length=MAX_QUESTION_LENGTH,
    doc_stride=DOC_STRIDE,
    cache_results=True)

#train_features = torch.load("./temp/cached_features_train")
#qa_examples = torch.load("./temp/cached_examples_train")

#print(len(train_features))
#print(train_features[0].input_ids)
# dev_features = torch.load("./temp/cached_features")
# dev_examples = torch.load("./temp/cached_examples")

#qa_extractor = BERTQAExtractor(language=LANGUAGE, cache_dir=CACHE_DIR, load_model_from_dir="./temp/distributed_0")


qa_results = qa_extractor.predict(dev_features, batch_size=BATCH_SIZE)

final_answers, answer_probs, nbest_answers = postprocess_answer(
     qa_results, dev_examples, dev_features, do_lower_case=DO_LOWER_CASE)

evaluation_result = evaluate_qa(
     qa_ids=dev_df["qa_id"], actuals=dev_df["answer_text"], preds=final_answers
)
