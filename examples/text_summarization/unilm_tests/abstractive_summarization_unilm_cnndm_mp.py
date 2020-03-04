import time
import datetime
import argparse

import torch.multiprocessing as mp
import torch

from utils_nlp.dataset.cnndm import CNNDMSummarizationDatasetOrg
from utils_nlp.models import S2SAbsSumProcessor, S2SAbstractiveSummarizer
from utils_nlp.eval import compute_rouge_python

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:29500")
parser.add_argument("--node_count", type=int, default=1)
parser.add_argument("--fp16", type=bool, default=False)
parser.add_argument("--fp16_opt_level", type=str, default="O2")


QUICK_RUN = True

OUTPUT_FILE = "./nlp_cnndm_finetuning_results.txt"

# model parameters
MODEL_NAME = "unilm-large-cased"
MAX_SEQ_LEN = 768
MAX_SOURCE_SEQ_LENGTH = 640
MAX_TARGET_SEQ_LENGTH = 128

# fine-tuning parameters
TRAIN_PER_GPU_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-5
if QUICK_RUN:
    TOP_N = 100
    WARMUP_STEPS = 10
    MAX_STEPS = 100
else:
    TOP_N = -1
    WARMUP_STEPS = 500
    MAX_STEPS = 5000

# inference parameters
TEST_PER_GPU_BATCH_SIZE = 24
BEAM_SIZE = 5
FORBID_IGNORE_WORD = "."


def main():
    start = time.time()
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()

    train_ds, test_ds = CNNDMSummarizationDatasetOrg(top_n=TOP_N)

    processor = S2SAbsSumProcessor(model_name=MODEL_NAME)
    abs_summarizer = S2SAbstractiveSummarizer(
        model_name=MODEL_NAME,
        max_seq_len=MAX_SEQ_LEN,
        max_source_seq_length=MAX_SOURCE_SEQ_LENGTH,
        max_target_seq_length=MAX_TARGET_SEQ_LENGTH,
    )

    train_dataset = processor.train_dataset_from_sum_ds(
        train_ds, load_cached_features=True
    )

    mp.spawn(
        main_worker,
        nprocs=ngpus_per_node,
        args=(
            ngpus_per_node,
            abs_summarizer,
            processor,
            train_dataset,
            args,
        ),
    )

def main_worker(
    local_rank,
    ngpus_per_node,
    abs_summarizer,
    processor,
    train_dataset,
    args,
):
    rank = args.rank * ngpus_per_node + local_rank
    world_size = args.node_count * ngpus_per_node

    print("init_method: {}".format(args.dist_url))
    print("ngpus_per_node: {}".format(ngpus_per_node))
    print("rank: {}".format(rank))
    print("local_rank: {}".format(local_rank))
    print("world_size: {}".format(world_size))

    torch.distributed.init_process_group(
        timeout=datetime.timedelta(0, 5400),
        backend="nccl",
        init_method=args.dist_url,
        world_size=world_size,
        rank=rank,
    )

    abs_summarizer.fit(
        train_dataset=train_dataset,
        per_gpu_batch_size=TRAIN_PER_GPU_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        local_rank=local_rank,
    )


if __name__ == "__main__":
    main()
