# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
probing tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import copy
import logging
import numpy as np

from senteval.tools.validation import SplitClassifier


class PROBINGEval(object):
    def __init__(self, task, task_path, seed=1111):
        self.seed = seed
        self.task = task
        logging.debug('***** (Probing) Transfer task : %s classification *****', self.task.upper())
        self.task_data = {'train': {'X': [], 'y': []},
                          'dev': {'X': [], 'y': []},
                          'test': {'X': [], 'y': []}}
        self.loadFile(task_path)
        logging.info('Loaded %s train - %s dev - %s test for %s' %
                     (len(self.task_data['train']['y']), len(self.task_data['dev']['y']),
                      len(self.task_data['test']['y']), self.task))

    def do_prepare(self, params, prepare):
        samples = self.task_data['train']['X'] + self.task_data['dev']['X'] + \
                  self.task_data['test']['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        self.tok2split = {'tr': 'train', 'va': 'dev', 'te': 'test'}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip().split('\t')
                self.task_data[self.tok2split[line[0]]]['X'].append(line[-1].split())
                self.task_data[self.tok2split[line[0]]]['y'].append(line[1])

        labels = sorted(np.unique(self.task_data['train']['y']))
        self.tok2label = dict(zip(labels, range(len(labels))))
        self.nclasses = len(self.tok2label)

        for split in self.task_data:
            for i, y in enumerate(self.task_data[split]['y']):
                self.task_data[split]['y'][i] = self.tok2label[y]

    def run(self, params, batcher):
        task_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size
        logging.info('Computing embeddings for train/dev/test')
        for key in self.task_data:
            # Sort to reduce padding
            sorted_data = sorted(zip(self.task_data[key]['X'],
                                     self.task_data[key]['y']),
                                 key=lambda z: (len(z[0]), z[1]))
            self.task_data[key]['X'], self.task_data[key]['y'] = map(list, zip(*sorted_data))

            task_embed[key]['X'] = []
            for ii in range(0, len(self.task_data[key]['y']), bsize):
                batch = self.task_data[key]['X'][ii:ii + bsize]
                embeddings = batcher(params, batch)
                task_embed[key]['X'].append(embeddings)
            task_embed[key]['X'] = np.vstack(task_embed[key]['X'])
            task_embed[key]['y'] = np.array(self.task_data[key]['y'])
        logging.info('Computed embeddings')

        config_classifier = {'nclasses': self.nclasses, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier}

        if self.task == "WordContent" and params.classifier['nhid'] > 0:
            config_classifier = copy.deepcopy(config_classifier)
            config_classifier['classifier']['nhid'] = 0
            print(params.classifier['nhid'])

        clf = SplitClassifier(X={'train': task_embed['train']['X'],
                                 'valid': task_embed['dev']['X'],
                                 'test': task_embed['test']['X']},
                              y={'train': task_embed['train']['y'],
                                 'valid': task_embed['dev']['y'],
                                 'test': task_embed['test']['y']},
                              config=config_classifier)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : %.1f Test acc : %.1f for %s classification\n' % (devacc, testacc, self.task.upper()))

        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(task_embed['dev']['X']),
                'ntest': len(task_embed['test']['X'])}

"""
Surface Information
"""
class LengthEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'sentence_length.txt')
        # labels: bins
        PROBINGEval.__init__(self, 'Length', task_path, seed)

class WordContentEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'word_content.txt')
        # labels: 200 target words
        PROBINGEval.__init__(self, 'WordContent', task_path, seed)

"""
Latent Structural Information
"""
class DepthEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'tree_depth.txt')
        # labels: bins
        PROBINGEval.__init__(self, 'Depth', task_path, seed)

class TopConstituentsEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'top_constituents.txt')
        # labels: 'PP_NP_VP_.' .. (20 classes)
        PROBINGEval.__init__(self, 'TopConstituents', task_path, seed)

class BigramShiftEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'bigram_shift.txt')
        # labels: 0 or 1
        PROBINGEval.__init__(self, 'BigramShift', task_path, seed)

# TODO: Voice?

"""
Latent Semantic Information
"""

class TenseEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'past_present.txt')
        # labels: 'PRES', 'PAST'
        PROBINGEval.__init__(self, 'Tense', task_path, seed)

class SubjNumberEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'subj_number.txt')
        # labels: 'NN', 'NNS'
        PROBINGEval.__init__(self, 'SubjNumber', task_path, seed)

class ObjNumberEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'obj_number.txt')
        # labels: 'NN', 'NNS'
        PROBINGEval.__init__(self, 'ObjNumber', task_path, seed)

class OddManOutEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'odd_man_out.txt')
        # labels: 'O', 'C'
        PROBINGEval.__init__(self, 'OddManOut', task_path, seed)

class CoordinationInversionEval(PROBINGEval):
    def __init__(self, task_path, seed=1111):
        task_path = os.path.join(task_path, 'coordination_inversion.txt')
        # labels: 'O', 'I'
        PROBINGEval.__init__(self, 'CoordinationInversion', task_path, seed)
