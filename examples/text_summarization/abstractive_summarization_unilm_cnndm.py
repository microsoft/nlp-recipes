import argparse
import torch.multiprocessing as mp

import torch

from utils_nlp.dataset.cnndm import CNNDMSummarizationDatasetOrg
from utils_nlp.models.transformers.abstractive_summarization_seq2seq import S2SAbsSumProcessor, S2SAbstractiveSummarizer

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:29500")
parser.add_argument("--node_count", type=int, default=1)

def main():
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()

    MODEL_NAME = "unilm-large-cased"

    train_ds, test_ds = CNNDMSummarizationDatasetOrg()

    processor = S2SAbsSumProcessor(model_name=MODEL_NAME)
    abs_summarizer = S2SAbstractiveSummarizer(
        model_name=MODEL_NAME,
        max_seq_len=768,
        max_source_seq_length=640,
        max_target_seq_length=128,
    )
    
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, abs_summarizer, processor, train_ds, args))


def main_worker(local_rank, ngpus_per_node, abs_summarizer, processor, train_ds, args):
    rank = args.rank * ngpus_per_node + local_rank
    world_size = args.node_count * ngpus_per_node

    PER_GPU_BATCH_SIZE = 2

    print("init_method: {}".format(args.dist_url))
    print("ngpus_per_node: {}".format(ngpus_per_node))
    print("rank: {}".format(rank))
    print("local_rank: {}".format(local_rank))
    print("world_size: {}".format(world_size))

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=world_size,
        rank=rank,
    )
    
    train_dataset = processor.train_dataset_from_sum_ds(train_ds, load_cached_features=True, local_rank=local_rank)

    abs_summarizer.fit(
        train_dataset=train_dataset,
        per_gpu_batch_size=PER_GPU_BATCH_SIZE,
        fp16=True,
        fp16_opt_level="O2",
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        warmup_steps=500,
        max_steps=5000,
        local_rank=local_rank
    )

if __name__ == "__main__":
    main()