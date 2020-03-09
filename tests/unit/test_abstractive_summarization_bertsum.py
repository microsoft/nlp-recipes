# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
torch.set_printoptions(threshold=5000)
from tempfile import TemporaryDirectory

from utils_nlp.models.transformers.datasets import (
    SummarizationDataset,
    SummarizationNonIterableDataset,
)
from utils_nlp.dataset.cnndm import CNNDMSummarizationDataset
from utils_nlp.models.transformers.abstractive_summarization_bertsum import BertSumAbs, BertSumAbsProcessor, validate
from utils_nlp.models.transformers.datasets import SummarizationNonIterableDataset
from utils_nlp.eval.evaluate_summarization import get_rouge

#CACHE_PATH = tmp_module #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"
#DATA_PATH = tmp_module  #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"
#MODEL_PATH = tmp_module #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"

# @pytest.fixture()
def source_data():
    return [
        [
            "Boston, MA welcome to Microsoft/nlp",
            "Welcome to text summarization.",
            "Welcome to Microsoft NERD.",
            "Look outside, what a beautiful Charlse River fall view."
        ],
        ["I am just another test case"],
        ["want to test more"],
    ]


# @pytest.fixture()
def target_data():
    return [
        ["welcome to microsoft/nlp.", "Welcome to text summarization.", "Welcome to Microsoft NERD."],
        ["I am just another test summary"],
        ["yest, I agree"],
    ]


NUM_GPUS = 2
os.environ["NCCL_IB_DISABLE"] = "0"

@pytest.fixture(scope="module")
def test_dataset_for_bertsumabs(tmp_module):
    CACHE_PATH = tmp_module #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"


    source = source_data()
    target = target_data()
    processor = BertSumAbsProcessor(cache_dir=CACHE_PATH)
    train_dataset = SummarizationNonIterableDataset(source, target)
    test_dataset = SummarizationNonIterableDataset(source, target)
    batch = processor.collate(train_dataset, 512, "cuda:0")
    assert len(batch.src) == 3
    return train_dataset, test_dataset

def shorten_dataset(dataset, top_n=-1):
    if top_n == -1:
        return dataset
    return SummarizationNonIterableDataset(dataset.source[0:top_n], dataset.target[0:top_n])

#def finetuned_model():
#    return os.path.join(MODEL_PATH, "dist_extsum_model.pt_step13000")
#    # return os.path.join(MODEL_PATH, "new_model_step_148000_torch1.4.0.pt")

"""
#def preprocess_cnndm_abs(top_n=8, need_process=True):
    if need_process is False:
        top_n = "full"
    train_data_path = os.path.join(DATA_PATH, "train_abssum_dataset_{}.pt".format(top_n))
    test_data_path = os.path.join(DATA_PATH, "test_abssum_dataset_{}.pt".format(top_n))
    if need_process:
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
        
        torch.save(train_sum_dataset, train_data_path)
        torch.save(test_sum_dataset, test_data_path)

    else:
        train_sum_dataset = torch.load(train_data_path)
        test_sum_dataset = torch.load(test_data_path)
    return train_sum_dataset, test_sum_dataset
"""

@pytest.mark.gpu
@pytest.fixture()
def test_train_model(tmp_module, test_dataset_for_bertsumabs, batch_size=1):
    CACHE_PATH = tmp_module #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"
    DATA_PATH = tmp_module  #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"
    MODEL_PATH = tmp_module #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"


    processor = BertSumAbsProcessor(cache_dir=CACHE_PATH)
    summarizer = BertSumAbs(
        processor, cache_dir=CACHE_PATH
    )
 
    checkpoint = None
    # train_sum_dataset, test_sum_dataset = preprocess_cnndm_abs(top_n=32)
    train_sum_dataset, test_sum_dataset = test_dataset_for_bertsumabs

    def this_validate(class_obj):
        return validate(class_obj, test_sum_dataset)

    MAX_STEP = 20
    TOP_N=8
    summarizer.fit(
        train_sum_dataset,
        batch_size=batch_size,
        max_steps=MAX_STEP,
        local_rank=-1,
        learning_rate_bert=0.002,
        learning_rate_dec=0.2,
        warmup_steps_bert=20000,
        warmup_steps_dec=10000,
        num_gpus=NUM_GPUS,
        report_every=10,
        save_every=100,
        validation_function=this_validate,
        fp16=False,
        fp16_opt_level="O1",
        checkpoint=checkpoint
    )
    saved_model_path = os.path.join(MODEL_PATH, "summarizer_step_{}.pt".format(MAX_STEP))
    summarizer.save_model(MAX_STEP, saved_model_path)

    src = test_sum_dataset.source[0:TOP_N]
    reference_summaries = ["".join(t).rstrip("\n") for t in test_sum_dataset.target[0:TOP_N]]
    generated_summaries = summarizer.predict(
        shorten_dataset(test_sum_dataset, top_n=TOP_N), batch_size=8
    )
    assert len(generated_summaries) == len(reference_summaries)
    RESULT_DIR = TemporaryDirectory().name
    rouge_score = get_rouge(generated_summaries, reference_summaries, RESULT_DIR)
    print(rouge_score)
    return saved_model_path

@pytest.mark.gpu
def test_finetuned_model(tmp_module, test_train_model, test_dataset_for_bertsumabs, top_n=8, batch_size=1, num_gpus=4):
    CACHE_PATH = tmp_module #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"
    DATA_PATH = tmp_module  #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"
    MODEL_PATH = tmp_module #"/dadendev/nlp-recipes/examples/text_summarization/abstemp"


    # train_sum_dataset, test_sum_dataset = preprocess_cnndm_abs(need_process=False)
    train_sum_dataset, test_sum_dataset = test_dataset_for_bertsumabs
    
    processor = BertSumAbsProcessor(cache_dir=CACHE_PATH)
    checkpoint = torch.load(test_train_model)
    
    summarizer = BertSumAbs(
        processor,
        cache_dir=CACHE_PATH,
        test=True,
        max_pos_length = 768
    )
    summarizer.model.load_checkpoint(checkpoint['model'])
    """
    summarizer.optim_bert = model_builder.build_optim_bert(
            summarizer.model,
            visible_gpus=None, #",".join([str(i) for i in range(num_gpus)]), #"0,1,2,3",
            lr_bert=0.002,
            warmup_steps_bert=20000,
            checkpoint=None,
        )
    summarizer.optim_dec = model_builder.build_optim_dec(
            summarizer.model,
            visible_gpus=None, #",".join([str(i) for i in range(num_gpus)]), #"0,1,2,3"
            lr_dec=0.2,
            warmup_steps_dec=10000,
        )
    summarizer.amp=None
    summarizer.save_model(20000, os.path.join(MODEL_PATH, "summarizer_step20000_with_global_step.pt"))
    return
    """
    top_n = 50 #len(test_sum_dataset) + 1 
    src = test_sum_dataset.source[0:top_n]
    reference_summaries = ["".join(t).rstrip("\n") for t in test_sum_dataset.target[0:top_n]]
    print("start prediction")
    generated_summaries = summarizer.predict(
        shorten_dataset(test_sum_dataset, top_n=top_n), batch_size=batch_size, num_gpus=num_gpus
    )
    def _write_list_to_file(list_items, filename):
        with open(filename, "w") as filehandle:
            # for cnt, line in enumerate(filehandle):
            for item in list_items:
                filehandle.write("%s\n" % item)

    print("writing generated summaries")
    _write_list_to_file(generated_summaries, os.path.join(CACHE_PATH, "prediction.txt"))

    assert len(generated_summaries) == len(reference_summaries)
    RESULT_DIR = TemporaryDirectory().name
    rouge_score = get_rouge(generated_summaries, reference_summaries, RESULT_DIR)
    print(rouge_score)


#test_preprocessing()
#preprocess_cnndm_abs()
#model_path = test_train_model()
#test_finetuned_model(model_path)
#test_finetuned_model(finetuned_model(),top_n=-1)

#if __name__ == "__main__":
#    main()
