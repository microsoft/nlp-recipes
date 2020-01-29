# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)


from utils_nlp.dataset.cnndm import CNNDMSummarizationDataset
from utils_nlp.models.transformers.extractive_summarization import (
    ExtractiveSummarizer,
    ExtSumProcessedData,
    ExtSumProcessor,
)

# os.environ["NCCL_BLOCKING_WAIT"] = "1"

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

os.environ["NCCL_IB_DISABLE"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:29500")
parser.add_argument("--node_count", type=int, default=1)
parser.add_argument("--cache_dir", type=str, default="./output")
parser.add_argument("--data_dir", type=str, default="/dadendev/temp_data5")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--quick_run", type=bool, default=False)
parser.add_argument("--use_preprocessed_data", type=bool, default=False)
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--top_n", type=int, default=3)

## Set QUICK_RUN = True to run the notebook
## on a small subset of data and a smaller number of epochs.


def cleanup():
    dist.destroy_process_group()


# Set USE_PREPROCSSED_DATA = True to skip the data preprocessing

# Transformer model being used

# notebook parameters
# the cache data path during find tuning
# batch size, unit is the number of tokens
BATCH_SIZE = 3000

# Encoder name. Options are: 1. baseline, classifier, transformer, rnn.
ENCODER = "transformer"

# How often the statistics reports show up in training, unit is step.
REPORT_EVERY = 100


def main():
    print("NCCL_SOCKET_IFNAME: {}".format(os.getenv("NCCL_SOCKET_IFNAME")))
    print("NCCL_DEBUG: {}".format(os.getenv("NCCL_DEBUG")))
    print("NCCL_DEBUG_SUBSYS: {}".format(os.getenv("NCCL_DEBUG_SUBSYS")))
    print("NCCL_IB_DISABLE: {}".format(os.getenv("NCCL_IB_DISABLE")))
    args = parser.parse_args()
    print("quick_run is {}".format(args.quick_run))
    print("use_preprocessed_data is {}".format(args.use_preprocessed_data))
    print("output_dir is {}".format(args.output_dir))
    print("output_dir is {}".format(args.output_dir))

    #shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    """
    def _write_list_to_file(list_items, filename):
        with open(filename, "w") as filehandle:
            # for cnt, line in enumerate(filehandle):
            for item in list_items:
                filehandle.write("%s\n" % item)

    print("writing generated summaries")
    _write_list_to_file(
        ["test1", "test2"], os.path.join(args.output_dir, "generated_summaries.txt")
    )
    import pickle

    favorite_color = {"lion": "yellow", "kitty": "red"}
    pickle.dump(favorite_color, open(os.path.join(args.output_dir, "save.p"), "wb"))
    """
    ngpus_per_node = torch.cuda.device_count()
    # sc = SequenceClassifier(model_name="bert-base-cased", num_labels=2, cache_dir=args.cache_dir)
    # sc.save_model()

    summarizer = ExtractiveSummarizer(args.model_name, ENCODER, args.cache_dir)
    #summarizer.save_model(os.path.join(args.output_dir, "dist_extsum_model.pt"))
    #return
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, summarizer, args))


def main_worker(local_rank, ngpus_per_node, summarizer, args):
    rank = args.rank * ngpus_per_node + local_rank
    world_size = args.node_count * ngpus_per_node

    print("init_method: {}".format(args.dist_url))
    print("ngpus_per_node: {}".format(ngpus_per_node))
    print("rank: {}".format(rank))
    print("local_rank: {}".format(local_rank))
    print("world_size: {}".format(world_size))

    torch.distributed.init_process_group(
        backend="nccl",
        # init_method="tcp://" + args.dist_url, # if "--dist_url": "$AZ_BATCH_MASTER_NODE"
        init_method=args.dist_url,
        world_size=world_size,
        rank=rank,
    )

    # summarizer.save_model(os.path.join(args.output_dir, "dis_sum_model.pt"))
    # return
    # setup(rank, args)
    # torch.cuda.set_device(rank)
    # if rank ==0:
    #    time.sleep(30)
    if args.use_preprocessed_data: #USE_PREPROCESSED_DATA:
        train_dataset, test_dataset = ExtSumProcessedData().splits(root=args.data_dir)
    else:
        save_path = os.path.join(args.data_dir, "processed")
        if rank ==0:
            DATA_PATH = args.data_dir  # TemporaryDirectory().name
            # The number of lines at the head of data file used for preprocessing.
            # -1 means all the lines.
            TOP_N = 1000
            CHUNK_SIZE = 200
            if not args.quick_run:
                TOP_N = -1
                CHUNK_SIZE = 2000
            train_raw_dataset, test_raw_dataset = CNNDMSummarizationDataset(
                top_n=TOP_N, local_cache_path=args.data_dir
            )
            processor = ExtSumProcessor(model_name=args.model_name)
            ext_sum_train = processor.preprocess(
                train_raw_dataset,
                train_raw_dataset.get_target(),
                oracle_mode="greedy",
                selections=args.top_n,
            )
            ext_sum_test = processor.preprocess(
                test_raw_dataset,
                test_raw_dataset.get_target(),
                oracle_mode="greedy",
                selections=args.top_n,
            )
            ExtSumProcessedData.save_data(
                ext_sum_train, is_test=False, save_path=save_path, chunk_size=CHUNK_SIZE
            )
            ExtSumProcessedData.save_data(
                ext_sum_test, is_test=True, save_path=save_path, chunk_size=CHUNK_SIZE
            )

        dist.barrier()    
        train_dataset, test_dataset = ExtSumProcessedData().splits(root=save_path)

    # total number of steps for training
    MAX_STEPS = 1e3
    # number of steps for warm up
    WARMUP_STEPS = 5e2
    if not args.quick_run:
        MAX_STEPS = 1e4
        WARMUP_STEPS = 1e3 * 5

    start = time.time()

    summarizer.fit(
        train_dataset,
        num_gpus=world_size,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        max_steps=MAX_STEPS / world_size,
        learning_rate=args.learning_rate,
        warmup_steps=WARMUP_STEPS,
        verbose=True,
        report_every=REPORT_EVERY,
        clip_grad_norm=False,
        local_rank=local_rank,
    )

    end = time.time()
    print("rank {0}, duration {1:.6f}s".format(rank, end - start))
    if rank in [-1, 0]:
        # summarizer.save_model(os.path.join(args.output_dir, "dis_sum_model.pt"))
        prediction = summarizer.predict(test_dataset, num_gpus=ngpus_per_node, batch_size=128)

        def _write_list_to_file(list_items, filename):
            with open(filename, "w") as filehandle:
                # for cnt, line in enumerate(filehandle):
                for item in list_items:
                    filehandle.write("%s\n" % item)

        print("writing generated summaries")
        _write_list_to_file(prediction, os.path.join(args.output_dir, "generated_summaries.txt"))
    # cleanup()


if __name__ == "__main__":
    main()
