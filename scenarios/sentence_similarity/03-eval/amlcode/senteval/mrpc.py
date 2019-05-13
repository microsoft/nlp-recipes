# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
MRPC : Microsoft Research Paraphrase (detection) Corpus
'''
from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import io

from senteval.tools.validation import KFoldClassifier

from sklearn.metrics import f1_score


class MRPCEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : MRPC *****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path,
                              'msr_paraphrase_train.txt'))
        test = self.loadFile(os.path.join(task_path,
                             'msr_paraphrase_test.txt'))
        self.mrpc_data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.mrpc_data['train']['X_A'] + \
                  self.mrpc_data['train']['X_B'] + \
                  self.mrpc_data['test']['X_A'] + self.mrpc_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        mrpc_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                mrpc_data['X_A'].append(text[3].split())
                mrpc_data['X_B'].append(text[4].split())
                mrpc_data['y'].append(text[0])

        mrpc_data['X_A'] = mrpc_data['X_A'][1:]
        mrpc_data['X_B'] = mrpc_data['X_B'][1:]
        mrpc_data['y'] = [int(s) for s in mrpc_data['y'][1:]]
        return mrpc_data

    def run(self, params, batcher):
        mrpc_embed = {'train': {}, 'test': {}}

        for key in self.mrpc_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.mrpc_data[key]['X_A'],
                                       self.mrpc_data[key]['X_B'],
                                       self.mrpc_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            text_data['A'] = [x for (x, y, z) in sorted_corpus]
            text_data['B'] = [y for (x, y, z) in sorted_corpus]
            text_data['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['A', 'B']:
                mrpc_embed[key][txt_type] = []
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    mrpc_embed[key][txt_type].append(embeddings)
                mrpc_embed[key][txt_type] = np.vstack(mrpc_embed[key][txt_type])
            mrpc_embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = mrpc_embed['train']['A']
        trainB = mrpc_embed['train']['B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = mrpc_embed['train']['y']

        # Test
        testA = mrpc_embed['test']['A']
        testB = mrpc_embed['test']['B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = mrpc_embed['test']['y']

        config = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid, 'kfold': params.kfold}
        clf = KFoldClassifier(train={'X': trainF, 'y': trainY},
                              test={'X': testF, 'y': testY}, config=config)

        devacc, testacc, yhat = clf.run()
        testf1 = round(100*f1_score(testY, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for MRPC.\n'
                      .format(devacc, testacc, testf1))
        return {'devacc': devacc, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainA), 'ntest': len(testA)}
