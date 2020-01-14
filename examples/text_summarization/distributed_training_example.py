import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
from torch.multiprocessing import Manager
from copy import deepcopy

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.distributed.init_process_group(
        backend="nccl",
        # init_method="tcp://" + args.dist_url, # if "--dist_url": "$AZ_BATCH_MASTER_NODE"
        #init_method=args.dist_url,
        init_method="tcp://"+os.environ['MASTER_ADDR']+":"+os.environ['MASTER_PORT'],
        world_size=world_size,
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

import os
import sys
from tempfile import TemporaryDirectory
import torch

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)


import json
import os
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import scrapbook as sb
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils_nlp.common.timer import Timer
from utils_nlp.dataset.multinli import load_pandas_df
from utils_nlp.models.transformers.sequence_classification import (
    Processor, SequenceClassifier)
from torch.utils.data import DataLoader

DATA_FOLDER = "/tmp/tmpmee459ec"#TemporaryDirectory().name
CACHE_DIR = TemporaryDirectory().name
NUM_EPOCHS = 1
BATCH_SIZE = 16
NUM_GPUS = 4
MAX_LEN = 100
TRAIN_DATA_FRACTION = 0.2
TEST_DATA_FRACTION = 0.05
TRAIN_SIZE = 0.75
LABEL_COL = "genre"
TEXT_COL = "sentence1"
MODEL_NAMES = ["distilbert-base-uncased"] # "roberta-base", "xlnet-base-cased"]

df = load_pandas_df(DATA_FOLDER, "train")
df = df[df["gold_label"]=="neutral"]  # get unique sentences

df_train, df_test = train_test_split(df, train_size = TRAIN_SIZE, random_state=0)
df_train = df_train.sample(frac=TRAIN_DATA_FRACTION).reset_index(drop=True)
df_test = df_test.sample(frac=TEST_DATA_FRACTION).reset_index(drop=True)
# encode labels
label_encoder = LabelEncoder()
df_train[LABEL_COL] = label_encoder.fit_transform(df_train[LABEL_COL])
df_test[LABEL_COL] = label_encoder.transform(df_test[LABEL_COL])

num_labels = len(np.unique(df_train[LABEL_COL]))

model_name = MODEL_NAMES[0]
# preprocess
processor = Processor(
    model_name=model_name,
    to_lower=model_name.endswith("uncased"),
    cache_dir=CACHE_DIR,
)

train_dataset = processor.dataset_from_dataframe(df_train, TEXT_COL, LABEL_COL, max_len=MAX_LEN)


# fine-tune
classifier = SequenceClassifier(
    model_name=model_name, num_labels=num_labels, cache_dir=CACHE_DIR
)



def score(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, collate_fn=None
        )

    start = time.time()
    classifier.fit(
            train_dataloader,
            num_epochs=NUM_EPOCHS,
            num_gpus=world_size,
            verbose=False,
            local_rank=rank
        )

    #prediction = summarizer.predict(test_dataset, num_gpus=world_size, local_rank=rank)
    end = time.time()
    print("rank {0}, duration {1:.6f}s".format(rank, end - start))
    cleanup()
    
def run_demo(demo_fn, world_size, ):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    run_demo(score, NUM_GPUS)
    classifier.save_model()
