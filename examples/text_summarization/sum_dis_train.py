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

def setup(rank, args):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://" + args.dist_url, # if "--dist_url": "$AZ_BATCH_MASTER_NODE"
        #init_method=args.dist_url,
        #init_method="tcp://"+os.environ['MASTER_ADDR']+":"+os.environ['MASTER_PORT'],
        world_size=args.world_size,
        rank=rank,
    )

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()

## Set QUICK_RUN = True to run the notebook on a small subset of data and a smaller number of epochs.
QUICK_RUN = False
## Set USE_PREPROCSSED_DATA = True to skip the data preprocessing
USE_PREPROCSSED_DATA = True


# Transformer model being used
MODEL_NAME = "distilbert-base-uncased"
PROCESSED_DATA_PATH = TemporaryDirectory().name
data_path = "./temp_data5/"
PROCESSED_DATA_PATH = data_path
if USE_PREPROCSSED_DATA:
    CNNDMBertSumProcessedData.download(local_path=PROCESSED_DATA_PATH)
    train_dataset, test_dataset = ExtSumProcessedData().splits(root=PROCESSED_DATA_PATH)
# notebook parameters
# the cache data path during find tuning
CACHE_DIR = './tmp2wg41gb5' #TemporaryDirectory().name

# batch size, unit is the number of tokens
BATCH_SIZE = 3000

# GPU used for training
NUM_GPUS = 4

# Encoder name. Options are: 1. baseline, classifier, transformer, rnn.
ENCODER = "transformer"


# Learning rate
LEARNING_RATE=1e-3

# How often the statistics reports show up in training, unit is step.
REPORT_EVERY=100

# total number of steps for training
MAX_STEPS=1e3
# number of steps for warm up
WARMUP_STEPS=5e2

if not QUICK_RUN:
    MAX_STEPS=1e4
    WARMUP_STEPS=1e3*5



def train(rank, args):
    world_size = args.world_size
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    start = time.time()

    summarizer = ExtractiveSummarizer(MODEL_NAME, ENCODER, CACHE_DIR)
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
            local_rank=rank,
        )

    end = time.time()
    print("rank {0}, duration {1:.6f}s".format(rank, end - start))
    if rank ==0 :
        summarizer.save_model("dis_sum_model.pt")
    cleanup()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dist-url', default='tcp://172.21.0.6:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--rank', default='0', type=str,
                    help='global ranks for the GPUS')

def run_demo(demo_fn, args, ):
    mp.spawn(demo_fn,
             args=(args,),
             nprocs=len(args.rank),
             join=True)

if __name__ == "__main__":
    args = parser.parse_args()
    args.rank = args.rank.split(",")
    run_demo(train, args)

