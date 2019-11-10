import argparse
import os
import sys
import jsonlines
import horovod.torch as hvd
import shutil

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.dataset.squad import load_pandas_df
from utils_nlp.dataset.pytorch import QADataset
from utils_nlp.models.transformers.question_answering import QAProcessor, AnswerExtractor
from utils_nlp.eval.question_answering import evaluate_qa
from utils_nlp.common.timer import Timer

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--rank", type=int, default=-1)
parser.add_argument("--dist_url", type=str, default="env://")
parser.add_argument("--node_count", type=int, default=1)
parser.add_argument("--cache_dir", type=str, default="./")
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
parser.add_argument("--do_lower_case", type=bool, default=True)
parser.add_argument("--quick_run", type=bool, default=False)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

args = parser.parse_args()

HOROVOD = True


hvd.init()

rank = hvd.rank()
local_rank = hvd.local_rank()
world_size = hvd.size()

print("rank: {}".format(rank))
print("local_rank: {}".format(local_rank))
print("world_size: {}".format(world_size))

MODEL_NAME = args.model_name
DO_LOWER_CASE = args.do_lower_case

TRAIN_DATA_USED_PERCENT = 1
DEV_DATA_USED_PERCENT = 1
NUM_EPOCHS = 2

MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
PER_GPU_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps

if args.quick_run:
    TRAIN_DATA_USED_PERCENT = 0.001
    DEV_DATA_USED_PERCENT = 0.01
    NUM_EPOCHS = 1

    MAX_SEQ_LENGTH = 128
    DOC_STRIDE = 64
    PER_GPU_BATCH_SIZE = 1

print("Max sequence length: {}".format(MAX_SEQ_LENGTH))
print("Document stride: {}".format(DOC_STRIDE))
print("Per gpu batch size: {}".format(PER_GPU_BATCH_SIZE))
print("Gradient accumulation steps: {}".format(GRADIENT_ACCUMULATION_STEPS))


RANDOM_SEED = 42
SQUAD_VERSION = "v1.1"
CACHE_DIR = args.cache_dir

MAX_QUESTION_LENGTH = 64
LEARNING_RATE = 3e-5

DOC_TEXT_COL = "doc_text"
QUESTION_TEXT_COL = "question_text"
ANSWER_START_COL = "answer_start"
ANSWER_TEXT_COL = "answer_text"
QA_ID_COL = "qa_id"
IS_IMPOSSIBLE_COL = "is_impossible"

train_df = load_pandas_df(
    local_cache_path=CACHE_DIR, squad_version=SQUAD_VERSION, file_split="train"
)
dev_df = load_pandas_df(local_cache_path=CACHE_DIR, squad_version=SQUAD_VERSION, file_split="dev")

train_df = train_df.sample(frac=TRAIN_DATA_USED_PERCENT).reset_index(drop=True)
dev_df = dev_df.sample(frac=DEV_DATA_USED_PERCENT).reset_index(drop=True)

train_dataset = QADataset(
    df=train_df,
    doc_text_col=DOC_TEXT_COL,
    question_text_col=QUESTION_TEXT_COL,
    qa_id_col=QA_ID_COL,
    is_impossible_col=IS_IMPOSSIBLE_COL,
    answer_start_col=ANSWER_START_COL,
    answer_text_col=ANSWER_TEXT_COL,
)
dev_dataset = QADataset(
    df=dev_df,
    doc_text_col=DOC_TEXT_COL,
    question_text_col=QUESTION_TEXT_COL,
    qa_id_col=QA_ID_COL,
    is_impossible_col=IS_IMPOSSIBLE_COL,
    answer_start_col=ANSWER_START_COL,
    answer_text_col=ANSWER_TEXT_COL,
)

qa_processor = QAProcessor(model_name=MODEL_NAME, to_lower=DO_LOWER_CASE)
train_features, _, _ = qa_processor.preprocess(
    train_dataset,
    is_training=True,
    max_question_length=MAX_QUESTION_LENGTH,
    max_seq_length=MAX_SEQ_LENGTH,
    doc_stride=DOC_STRIDE,
)

dev_features, qa_examples_json, features_json = qa_processor.preprocess(
    dev_dataset,
    is_training=False,
    max_question_length=MAX_QUESTION_LENGTH,
    max_seq_length=MAX_SEQ_LENGTH,
    doc_stride=DOC_STRIDE,
)

print("preprocessing finished")

if local_rank in [-1, 0]:

    feature_cache_dir = "./cached_qa_features"
    CACHED_EXAMPLES_TEST_FILE = "cached_examples_test.jsonl"
    CACHED_FEATURES_TEST_FILE = "cached_features_test.jsonl"
    examples_file = os.path.join(feature_cache_dir, CACHED_EXAMPLES_TEST_FILE)
    features_file = os.path.join(feature_cache_dir, CACHED_FEATURES_TEST_FILE)

    if os.path.isdir(feature_cache_dir):
        shutil.rmtree(feature_cache_dir, ignore_errors=True)
    os.mkdir(feature_cache_dir)

    with jsonlines.open(examples_file, "w") as examples_writer, jsonlines.open(
        features_file, "w"
    ) as features_writer:

        examples_writer.write_all(qa_examples_json)
        features_writer.write_all(features_json)

    print("features cahed")

qa_extractor = AnswerExtractor(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
print("model loaded")

with Timer() as t:
    qa_extractor.fit(
        train_dataset=train_features,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_gpu_batch_size=PER_GPU_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        seed=RANDOM_SEED,
        cache_model=True,
        local_rank=local_rank,
        world_size=world_size,
        rank=rank,
        hvd_dist=HOROVOD,
        weight_decay=0.01,
        adam_epsilon=1e-6,
        warmup_steps=274,
    )
print("Training time : {:.3f} hrs".format(t.interval / 3600))


if local_rank in [-1, 0]:
    qa_results = qa_extractor.predict(dev_features, per_gpu_batch_size=PER_GPU_BATCH_SIZE)
    final_answers, answer_probs, nbest_answers = qa_processor.postprocess(
        qa_results,
        examples_file="./cached_qa_features/cached_examples_test.jsonl",
        features_file="./cached_qa_features/cached_features_test.jsonl",
    )

    evaluation_result = evaluate_qa(actual_dataset=dev_dataset, preds=final_answers)
