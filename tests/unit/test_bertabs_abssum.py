# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys

sys.path.insert(0, "../../")

import argparse
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tempfile import TemporaryDirectory

from utils_nlp.models.transformers.datasets import (
    SummarizationDataset,
    SummarizationNonIterableDataset,
)
from utils_nlp.models.transformers.abssum import AbsSum, AbsSumProcessor, validate

from utils_nlp.dataset.cnndm import CNNDMBertSumProcessedData, CNNDMSummarizationDataset
from utils_nlp.models.transformers.datasets import SummarizationNonIterableDataset
from utils_nlp.eval.evaluate_summarization import get_rouge

CACHE_PATH = "/dadendev/nlp-recipes/examples/text_summarization/abstemp"
DATA_PATH = "/dadendev/nlp-recipes/examples/text_summarization"
MODEL_PATH = "/dadendev/nlp-recipes/examples/text_summarization/abstemp"
TOP_N = 10

# @pytest.fixture()
def source_data():
    return [
        (
            "Boston, MA welcome to Microsoft/nlp. Welcome to text summarization."
            "Welcome to Microsoft NERD."
            "Look outside, what a beautiful Charlse River fall view."
        ),
        ("I am just another test case"),
        ("want to test more"),
    ]


# @pytest.fixture()
def target_data():
    return [
        ("welcome to microsoft/nlp." "Welcome to text summarization." "Welcome to Microsoft NERD."),
        ("I am just another test summary"),
        ("yest, I agree"),
    ]


MODEL_NAME = "distilbert-base-uncased"
NUM_GPUS = 1
os.environ["NCCL_IB_DISABLE"] = "0"

def test_preprocessing():
    source = source_data()
    target = target_data()
    print(source)
    print(target)
    processor = AbsSumProcessor()
    train_dataset = SummarizationNonIterableDataset(source, target)
    batch = processor.collate(train_dataset, 512, "cuda:0")
    print(batch)
    print(len(batch.src[0]))


def shorten_dataset(dataset, top_n=-1):
    if top_n == -1:
        return dataset
    return SummarizationNonIterableDataset(dataset.source[0:top_n], dataset.target[0:top_n])


# @pytest.fixture(scope="module")
def pretrained_model():
    return torch.load(os.path.join(MODEL_PATH, "model_step_148000_torch1.4.0.pt"))

def preprocess_cnndm_abs():
    #TOP_N = -1
    train_data_path = os.path.join(DATA_PATH, "train_abssum_dataset_full.pt")
    test_data_path = os.path.join(DATA_PATH, "test_abssum_dataset_full.pt")
    if False:
        print("processing data")
        train_dataset, test_dataset = CNNDMSummarizationDataset(
            top_n=TOP_N, local_cache_path=DATA_PATH, prepare_extractive=False
        )
        source = [x[0] for x in list(test_dataset.get_source())]
        target = [x[0] for x in list(test_dataset.get_target())]
        test_sum_dataset = SummarizationNonIterableDataset(source, target)

        source = [x[0] for x in list(train_dataset.get_source())]
        target = [x[0] for x in list(train_dataset.get_target())]
        train_sum_dataset = SummarizationNonIterableDataset(source, target)
        
        if TOP_N == -1:
            torch.save(train_sum_dataset, train_data_path)
            torch.save(test_sum_dataset, test_data_path)

    else:
        train_sum_dataset = torch.load(train_data_path)
        test_sum_dataset = torch.load(test_data_path)
    return train_sum_dataset, test_sum_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=0,
                    help="The rank of the current node in the cluster")
parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:29501",
                    help="URL specifying how to initialize the process groupi.")

parser.add_argument("--node_count", type=int, default=1,
                    help="Number of nodes in the cluster.")
def main():

    #shutil.rmtree(args.output_dir)
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    processor = AbsSumProcessor(cache_dir=CACHE_PATH)
    checkpoint = torch.load(os.path.join(MODEL_PATH, "summarizer_step20000.pt"))
    checkpoint = None
    summarizer = AbsSum(
        processor, checkpoint=checkpoint, cache_dir=CACHE_PATH
    )


    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, summarizer,  args))


def main_worker(local_rank, ngpus_per_node, summarizer, args):
    rank = args.rank * ngpus_per_node + local_rank
    world_size = args.node_count * ngpus_per_node
    print("world_size is {}".format(world_size))
    print("local_rank is {} and rank is {}".format(local_rank, rank))

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=world_size,
        rank=rank,
      )
    torch.cuda.set_device(local_rank)
    #return  
    train_sum_dataset, test_sum_dataset = preprocess_cnndm_abs()
    def this_validate(class_obj):
        return validate(class_obj, os.path.join(DATA_PATH, "test_abssum_dataset_full.pt"), CACHE_PATH)

    if rank not in [-1, 0]:
        save_every = -1
        this_validate = None
    else:
        save_every = 10
        this_validate = None

    summarizer.fit(
        train_sum_dataset,
        world_size=world_size,
        num_gpus=None,
        local_rank=local_rank,
        rank=rank,
        batch_size=2,
        max_steps=30000,
        learning_rate_bert=0.002,
        warmup_steps_bert=20000,
        warmup_steps_dec=10000,
        save_every=save_every,
        report_every=10,
        validation_function=this_validate,
        fp16=False
    )

    dist.destroy_process_group()

    

def test_train_model():
    processor = AbsSumProcessor(cache_dir=CACHE_PATH)
    checkpoint = torch.load(os.path.join(MODEL_PATH, "summarizer_step20000.pt"))
    # checkpoint = torch.load(os.path.join(MODEL_PATH, "bert-base-uncased_step_2900.pt"))
    #checkpoint = None
    summarizer = AbsSum(
        processor, checkpoint=None, cache_dir=CACHE_PATH
    )
 
    summarizer.model.load_checkpoint(checkpoint['model'])
   
    train_sum_dataset, test_sum_dataset = preprocess_cnndm_abs()
   
    def this_validate(class_obj):
        return validate(class_obj, os.path.join(DATA_PATH, "test_abssum_dataset_full.pt"), CACHE_PATH)
    summarizer.fit(
        train_sum_dataset,
        batch_size=8,
        max_steps=30000,
        learning_rate_bert=0.002,
        warmup_steps_bert=20000,
        warmup_steps_dec=10000,
        num_gpus=2,
        save_every=100,
        report_every=10,
        validation_function=this_validate,
        fp16=False
    )
    saved_model_path = os.path.join(MODEL_PATH, "summarizer_step50000.pt")
    summarizer.save_model(saved_model_path)

    summarizer = AbsSum(
        processor,
        checkpoint=torch.load(saved_model_path),
        cache_dir=CACHE_PATH,
    )

    src = test_sum_dataset.source[0:TOP_N]
    reference_summaries = ["".join(t).rstrip("\n") for t in test_sum_dataset.target[0:TOP_N]]
    generated_summaries = summarizer.predict(
        shorten_dataset(test_sum_dataset, top_n=TOP_N), batch_size=8
    )
    assert len(generated_summaries) == len(reference_summaries)
    for i in generated_summaries:
        print(i)
        print("\n")
        print("###################")

    for i in reference_summaries:
        print(i)
        print("\n")

    RESULT_DIR = TemporaryDirectory().name
    rouge_score = get_rouge(generated_summaries, reference_summaries, RESULT_DIR)
    print(rouge_score)


def test_pretrained_model():
    train_sum_dataset, test_sum_dataset = preprocess_cnndm_abs()
    
    processor = AbsSumProcessor(cache_dir=CACHE_PATH)
    checkpoint = torch.load(os.path.join(MODEL_PATH, "model_step_148000_torch1.4.0.pt"))
    #checkpoint = torch.load(os.path.join(MODEL_PATH, "bert-base-uncased_step_2900.pt"))
    #checkpoint = torch.load(os.path.join(MODEL_PATH, "summarizer_step20000.pt"))
    summarizer = AbsSum(
        processor,
        checkpoint=None,
        cache_dir=CACHE_PATH,
    )
    summarizer.model.load_checkpoint(checkpoint['model'])
    
    top_n = 10
    src = test_sum_dataset.source[0:top_n]
    reference_summaries = ["".join(t).rstrip("\n") for t in test_sum_dataset.target[0:top_n]]
    print("start prediction")
    generated_summaries = summarizer.predict(
        shorten_dataset(test_sum_dataset, top_n=top_n), batch_size=3, num_gpus=2
    )
    assert len(generated_summaries) == len(reference_summaries)
    RESULT_DIR = TemporaryDirectory().name
    rouge_score = get_rouge(generated_summaries, reference_summaries, RESULT_DIR)
    print(rouge_score)
    assert rouge_score["rouge_2_f_score"] > 0.17


#test_preprocessing()
#preprocess_cnndm_abs()
test_train_model()
#test_pretrained_model()
#if __name__ == "__main__":
#    main()
