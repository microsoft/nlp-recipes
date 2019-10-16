# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
sys.path.insert(0, "/dadendev/nlp/")
import pytest
import os
import shutil
from utils_nlp.dataset.harvardnlp_cnndm import harvardnlp_cnndm_preprocess
from utils_nlp.models.bert.extractive_text_summarization import bertsum_formatting

from bertsum.prepro.data_builder import BertData
from utils_nlp.models.bert.extractive_text_summarization import Bunch, BertSumExtractiveSummarizer, get_data_iter

import urllib.request

#@pytest.fixture()
def source_data():
    return """boston, MA -lrb- msft -rrb- welcome to Microsoft/nlp. Welcome to text summarization. Welcome to Microsoft NERD. Look out, beautiful Charlse River fall view."""
#@pytest.fixture()
def target_data():
    return """<t> welcome to microsfot/nlp. </t> <t>  Welcome to text summarization.</t> <t> Welcome to Microsoft NERD.</t> """

@pytest.fixture()
def bertdata_file():
    source= source_data()
    target = target_data()
    source_file = "source.txt"
    target_file = "target.txt"
    bertdata_file = "bertdata"
    f = open(source_file, "w")
    f.write(source)
    f.close()
    f = open(target_file, "w")
    f.write(target)
    f.close()
    jobs = harvardnlp_cnndm_preprocess(1, source_file, target_file, 2)
    assert len(jobs) == 1
    default_preprocessing_parameters =  {"max_nsents": 200, "max_src_ntokens": 2000, "min_nsents": 3, "min_src_ntokens": 2, "use_interval": True}
    args=Bunch(default_preprocessing_parameters)
    bertdata = BertData(args)
    bertsum_formatting(1, bertdata,"combination", jobs, bertdata_file)
    assert os.path.exists(bertdata_file)
    os.remove(source_file)
    os.remove(target_file)
    return bertdata_file

@pytest.mark.gpu
def test_training(bertdata_file):
    device_id = 0
    gpu_ranks = str(device_id)

    BERT_CONFIG_PATH="./bert_config_uncased_base.json"

    filedata = urllib.request.urlretrieve('https://raw.githubusercontent.com/nlpyang/BertSum/master/bert_config_uncased_base.json', BERT_CONFIG_PATH)

    encoder = 'transformer'
    model_base_path = './models/'
    log_base_path = './logs/'
    result_base_path = './results'

    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)
    if not os.path.exists(log_base_path):
        os.makedirs(log_base_path)
    if not os.path.exists(result_base_path):
        os.makedirs(result_base_path)



    from random import random
    random_number = random()
    import torch
    #bertdata_file = "bertdata"
    data = torch.load(bertdata_file)
    assert len(data) == 1
    bertsum_model = BertSumExtractiveSummarizer(encoder = encoder,
                                            model_path = model_base_path + encoder + str(random_number),
                                            log_file = log_base_path + encoder + str(random_number),
                                            bert_config_path = BERT_CONFIG_PATH,
                                            gpu_ranks = gpu_ranks,)
    bertsum_model.args.save_checkpoint_steps = 50
    train_steps = 100
    bertsum_model.fit(device_id, [bertdata_file], train_steps=train_steps, train_from="")
    model_for_test =  os.path.join(model_base_path + encoder + str(random_number), f"model_step_{train_steps}.pt")
    assert os.path.exists(model_for_test)
    prediction = bertsum_model.predict(device_id, get_data_iter(data),
                                   test_from=model_for_test,
                                   sentence_seperator='<q>')
    assert len(prediction) == 1
    if os.path.exists(model_base_path):
        shutil.rmtree(model_base_path)
    if os.path.exists(log_base_path):
        shutil.rmtree(log_base_path)
    if os.path.exists(result_base_path):
        shutil.rmtree(result_base_path)
    if os.path.isfile(BERT_CONFIG_PATH):
        os.remove(BERT_CONFIG_PATH)
    if os.path.isfile(bertdata_file):
        os.remove(bertdata_file)
