# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Clone GenSen repo here: https://github.com/Maluuba/gensen.git
And follow instructions for loading the model used in batcher
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import logging
# import GenSen package
from gensen import GenSen, GenSenSingle

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    _, reps_h_t = gensen.get_representation(
        sentences, pool='last', return_numpy=True, tokenize=True
    )
    embeddings = reps_h_t
    return embeddings

# Load GenSen model
gensen_1 = GenSenSingle(
    model_folder='../data/models',
    filename_prefix='nli_large_bothskip',
    pretrained_emb='../data/embedding/glove.840B.300d.h5'
)
gensen_2 = GenSenSingle(
    model_folder='../data/models',
    filename_prefix='nli_large_bothskip_parse',
    pretrained_emb='../data/embedding/glove.840B.300d.h5'
)
gensen_encoder = GenSen(gensen_1, gensen_2)
reps_h, reps_h_t = gensen.get_representation(
    sentences, pool='last', return_numpy=True, tokenize=True
)

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
params_senteval['gensen'] = gensen_encoder

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
