import argparse
import datetime
import distutils
import os
import sys
from tempfile import TemporaryDirectory
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from torch.multiprocessing import Manager
from copy import deepcopy

#os.environ["NCCL_BLOCKING_WAIT"] = "1"

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

os.environ["NCCL_IB_DISABLE"] = "0"

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)


nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.common.pytorch_utils import get_device
from utils_nlp.dataset.cnndm import CNNDMBertSumProcessedData
from utils_nlp.eval.evaluate_summarization import get_rouge
from utils_nlp.models.transformers.extractive_summarization import (
    ExtractiveSummarizer,
    ExtSumProcessedData,
    ExtSumProcessor,
)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:29500")
parser.add_argument("--node_count", type=int, default=1)
parser.add_argument("--cache_dir", type=str, default="./")
parser.add_argument("--data_dir", type=str, default="./")
parser.add_argument("--output_dir", type=str, default="./")
parser.add_argument("--quick_run", type=bool, default=False)


def setup(rank, args):

    # initialize the process group
    print("world size {}".format(args.world_size))
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url, # if "--dist_url": "$AZ_BATCH_MASTER_NODE"
        timeout=datetime.timedelta(0, 1),
        world_size=args.world_size,
        rank=rank,
    )
    # start from same random weights and biases.
    torch.manual_seed(42)
    print("world size is {}".format(torch.distributed.get_world_size()))


def cleanup():
    dist.destroy_process_group()

## Set QUICK_RUN = True to run the notebook on a small subset of data and a smaller number of epochs.
## Set USE_PREPROCSSED_DATA = True to skip the data preprocessing
USE_PREPROCSSED_DATA = True

# Transformer model being used
MODEL_NAME = "distilbert-base-uncased"
    
# notebook parameters
# the cache data path during find tuning
#CACHE_DIR = './tmp2wg41gb5' #TemporaryDirectory().name

# batch size, unit is the number of tokens
BATCH_SIZE = 3000

# Encoder name. Options are: 1. baseline, classifier, transformer, rnn.
ENCODER = "transformer"

# Learning rate
LEARNING_RATE=1e-3

# How often the statistics reports show up in training, unit is step.
REPORT_EVERY=100


def main():
    print("NCCL_SOCKET_IFNAME: {}".format(os.getenv("NCCL_SOCKET_IFNAME")))
    print("NCCL_DEBUG: {}".format(os.getenv("NCCL_DEBUG")))
    print("NCCL_DEBUG_SUBSYS: {}".format(os.getenv("NCCL_DEBUG_SUBSYS")))
    print("NCCL_IB_DISABLE: {}".format(os.getenv("NCCL_IB_DISABLE")))
    args = parser.parse_args()
    print("quick run is {}".format(args.quick_run))
    print("output_dir is {}".format(args.output_dir))
    print("output_dir is {}".format(args.output_dir))

    def _write_list_to_file(list_items, filename):
            with open(filename, "w") as filehandle:
                # for cnt, line in enumerate(filehandle):
                for item in list_items:
                    filehandle.write("%s\n" % item)
    print("writing generated summaries")
    _write_list_to_file(["test1", "test2"], os.path.join(args.output_dir, "generated_summaries.txt"))
    _write_list_to_file(["test1", "test2"], os.path.join(args.output_dir, "dist_extsum_model.pt"))
    ngpus_per_node = torch.cuda.device_count()

    #MODEL_NAME = args.model_name
    summarizer = ExtractiveSummarizer(MODEL_NAME, ENCODER, args.cache_dir)

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

    #summarizer.save_model(os.path.join(args.output_dir, "dis_sum_model.pt"))
    #return
    #setup(rank, args)
    #torch.cuda.set_device(rank)
    #if rank ==0:
    #    time.sleep(30)
    train_dataset, test_dataset = ExtSumProcessedData().splits(root=args.data_dir)
    # total number of steps for training
    MAX_STEPS=1e3
    # number of steps for warm up
    WARMUP_STEPS=5e2
    if not args.quick_run:
        MAX_STEPS=1e4
        WARMUP_STEPS=1e3*5

    start = time.time()

    summarizer.fit(
            train_dataset,
            num_gpus=world_size,
            batch_size=BATCH_SIZE,
            gradient_accumulation_steps=2,
            max_steps=MAX_STEPS/world_size,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            verbose=True,
            report_every=REPORT_EVERY,
            clip_grad_norm=False,
            local_rank=local_rank,
        )

    end = time.time()
    print("rank {0}, duration {1:.6f}s".format(rank, end - start))
    if rank in [-1,0]:
        #summarizer.save_model(os.path.join(args.output_dir, "dis_sum_model.pt"))
        prediction = summarizer.predict(test_dataset, num_gpus=ngpus_per_node, batch_size=128)
        
        def _write_list_to_file(list_items, filename):
            with open(filename, "w") as filehandle:
                # for cnt, line in enumerate(filehandle):
                for item in list_items:
                    filehandle.write("%s\n" % item)
        print("writing generated summaries")
        _write_list_to_file(prediction, os.path.join(args.output_dir, "generated_summaries.txt"))
        #target = [i['tgt_txt'] for i in test_dataset]
        #rouge_score = get_rouge(prediction, target, args.cache_dir)
        #print(rouge_score)
    #cleanup()


if __name__ == "__main__":
    main()
