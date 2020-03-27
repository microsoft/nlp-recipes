import datetime
import argparse
import jsonlines

import torch

from utils_nlp.models.transformers.abstractive_summarization_seq2seq import (
     S2SAbsSumProcessor, 
     S2SAbstractiveSummarizer
)

from utils_nlp.eval import compute_rouge_python

parser = argparse.ArgumentParser()
parser.add_argument(
    "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
)
parser.add_argument("--fp16", type=bool, default=False)
parser.add_argument("--fp16_opt_level", type=str, default="O2")
args = parser.parse_args()


QUICK_RUN = True
OUTPUT_FILE = "./nlp_cnndm_finetuning_results.txt"

# model parameters
MODEL_NAME = "unilm-large-cased"
MAX_SEQ_LENGTH = 768
MAX_SOURCE_SEQ_LENGTH = 640
MAX_TARGET_SEQ_LENGTH = 128

# fine-tuning parameters
TRAIN_PER_GPU_BATCH_SIZE = 1
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
TEST_PER_GPU_BATCH_SIZE = 8
BEAM_SIZE = 5
FORBID_IGNORE_WORD = "."

train_ds = "train_ds.jsonl"
test_ds = "test_ds.jsonl"


def main():
    torch.distributed.init_process_group(
        timeout=datetime.timedelta(0, 5400), backend="nccl",
    )

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = S2SAbsSumProcessor(model_name=MODEL_NAME)

    abs_summarizer = S2SAbstractiveSummarizer(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        max_source_seq_length=MAX_SOURCE_SEQ_LENGTH,
        max_target_seq_length=MAX_TARGET_SEQ_LENGTH,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_dataset = processor.s2s_dataset_from_json_or_file(
        train_ds, train_mode=True, local_rank=args.local_rank
    )

    test_dataset = processor.s2s_dataset_from_json_or_file(
        test_ds, train_mode=False, local_rank=args.local_rank
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
        local_rank=args.local_rank,
        save_model_to_dir=".",
    )

    torch.distributed.barrier()

    if args.local_rank in [-1, 0]:
        res = abs_summarizer.predict(
            test_dataset=test_dataset,
            per_gpu_batch_size=TEST_PER_GPU_BATCH_SIZE,
            beam_size=BEAM_SIZE,
            forbid_ignore_word=FORBID_IGNORE_WORD,
            fp16=args.fp16,
        )

        for r in res[:5]:
            print(r)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for line in res:
                f.write(line + "\n")

        tgt = []
        with jsonlines.open(test_ds) as reader:
            for item in reader:
                tgt.append(item["tgt"])

        for t in tgt[:5]:
            print(t)

        print(compute_rouge_python(cand=res, ref=tgt))


if __name__ == "__main__":
    main()
