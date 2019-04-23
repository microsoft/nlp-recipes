#!/usr/bin/env python
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# In addition to the legal release guidance under MIT please note in this file
# inspired by https://github.com/facebookresearch/SentEval/blob/master/examples/infersent.py 
# that portions of the code are covered by this license: https://github.com/facebookresearch/SentEval/blob/master/LICENSE

from __future__ import absolute_import, division, unicode_literals

import sys
sys.path.append('.')
import torch
import logging

import argparse
from gensen import GenSen, GenSenSingle

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

# set gpu device
torch.cuda.set_device(0)


def prepare(params, samples):
    print('Preparing task : %s ' % (params.current_task))
    vocab = set()
    for sample in samples:
        if params.current_task != 'TREC':
            sample = ' '.join(sample).lower().split()
        else:
            sample = ' '.join(sample).split()
        for word in sample:
            if word not in vocab:
                vocab.add(word)

    vocab.add('<s>')
    vocab.add('<pad>')
    vocab.add('<unk>')
    vocab.add('</s>')
    # If you want to turn off vocab expansion just comment out the below line.
    params['gensen'].vocab_expansion(vocab)


def batcher(params, batch):
    # batch contains list of words
    max_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'ImageCaptionRetrieval']
    if args.strategy == 'best':
        if params.current_task in max_tasks:
            strategy = 'max'
        else:
            strategy = 'last'
    else:
        strategy = args.strategy

    sentences = [' '.join(s).lower() for s in batch]
    _, embeddings = params['gensen'].get_representation(
        sentences, pool=strategy, return_numpy=True
    )
    return embeddings

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'SICKRelatedness',\
                  'SICKEntailment', 'MRPC', 'STS14', 'STSBenchmark', 'STS12', 'STS13', 'STS15', 'STS16']
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

if __name__ == "__main__":
    # Load model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path",
        help="path to model folder",
        default='./data/models'
    )
    parser.add_argument(
        "--prefix_1",
        help="prefix to model 1",
        default='nli_large_bothskip_parse'
    )
    parser.add_argument(
        "--prefix_2",
        help="prefix to model 2",
        default='nli_large_bothskip'
    )
    parser.add_argument(
        "--pretrain",
        help="path to pretrained vectors",
        default='./data/embedding/glove.840B.300d.h5'
    )
    parser.add_argument(
        "--strategy",
        help="Approach to create sentence embedding last/max/best",
        default="best",  # NOTE: To decide the pooling strategy for a new model, note down the validation set scores below.
    )
    parser.add_argument(
        "--cuda",
        help="Use GPU to compute sentence representations",
        default=torch.cuda.is_available()
    )
    args = parser.parse_args()

    print('#############################')
    print('####### Parameters ##########')
    print('Prefix 1 : %s ' % (args.prefix_1))
    print('Prefix 2 : %s ' % (args.prefix_2))
    print('Pretrained Embeddings : %s ' % (args.pretrain))
    print('#############################')

    gensen_1 = GenSenSingle(
        model_folder=args.folder_path,
        filename_prefix=args.prefix_1,
        pretrained_emb=args.pretrain,
        cuda=args.cuda
    )
    gensen_2 = GenSenSingle(
        model_folder=args.folder_path,
        filename_prefix=args.prefix_2,
        pretrained_emb=args.pretrain,
        cuda=args.cuda
    )
    gensen = GenSen(gensen_1, gensen_2)
    params_senteval['gensen'] = gensen
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print('--------------------------------------------')
    print('Table 2 of Our Paper : ')
    print('--------------------------------------------')
    print('MR                [Dev:%.1f/Test:%.1f]' % (results_transfer['MR']['devacc'], results_transfer['MR']['acc']))
    print('CR                [Dev:%.1f/Test:%.1f]' % (results_transfer['CR']['devacc'], results_transfer['CR']['acc']))
    print('SUBJ              [Dev:%.1f/Test:%.1f]' % (results_transfer['SUBJ']['devacc'], results_transfer['SUBJ']['acc']))
    print('MPQA              [Dev:%.1f/Test:%.1f]' % (results_transfer['MPQA']['devacc'], results_transfer['MPQA']['acc']))
    print('SST2              [Dev:%.1f/Test:%.1f]' % (results_transfer['SST2']['devacc'], results_transfer['SST2']['acc']))
    print('SST5              [Dev:%.1f/Test:%.1f]' % (results_transfer['SST5']['devacc'], results_transfer['SST5']['acc']))
    print('TREC              [Dev:%.1f/Test:%.1f]' % (results_transfer['TREC']['devacc'], results_transfer['TREC']['acc']))
    print('MRPC              [Dev:%.1f/TestAcc:%.1f/TestF1:%.1f]' % (results_transfer['MRPC']['devacc'], results_transfer['MRPC']['acc'], results_transfer['MRPC']['f1']))
    print('SICKRelatedness   [Dev:%.3f/Test:%.3f]' % (results_transfer['SICKRelatedness']['devpearson'], results_transfer['SICKRelatedness']['pearson']))
    print('SICKEntailment    [Dev:%.1f/Test:%.1f]' % (results_transfer['SICKEntailment']['devacc'], results_transfer['SICKEntailment']['acc']))
    print('STS12             [Pearson:%.3f/Spearman:%.3f]' % (results_transfer['STS12']['all']['pearson']['mean'], results_transfer['STS12']['all']['spearman']['mean']))
    print('STS13             [Pearson:%.3f/Spearman:%.3f]' % (results_transfer['STS13']['all']['pearson']['mean'], results_transfer['STS13']['all']['spearman']['mean']))
    print('STS14             [Pearson:%.3f/Spearman:%.3f]' % (results_transfer['STS14']['all']['pearson']['mean'], results_transfer['STS14']['all']['spearman']['mean']))
    print('STS15             [Pearson:%.3f/Spearman:%.3f]' % (results_transfer['STS15']['all']['pearson']['mean'], results_transfer['STS15']['all']['spearman']['mean']))
    print('STS16             [Pearson:%.3f/Spearman:%.3f]' % (results_transfer['STS16']['all']['pearson']['mean'], results_transfer['STS16']['all']['spearman']['mean']))
    print('STSBenchmark      [Dev:%.5f/Pearson:%.5f/Spearman:%.5f]' % (results_transfer['STSBenchmark']['devpearson'], results_transfer['STSBenchmark']['pearson'], results_transfer['STSBenchmark']['spearman']))
    print('--------------------------------------------')
