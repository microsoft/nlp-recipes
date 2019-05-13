# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Image-Caption Retrieval with COCO dataset
'''
from __future__ import absolute_import, division, unicode_literals

import os
import sys
import logging
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

from senteval.tools.ranking import ImageSentenceRankingPytorch


class ImageCaptionRetrievalEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task: Image Caption Retrieval *****\n\n')

        # Get captions and image features
        self.seed = seed
        train, dev, test = self.loadFile(task_path)
        self.coco_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.coco_data['train']['sent'] + \
                  self.coco_data['dev']['sent'] + \
                  self.coco_data['test']['sent']
        prepare(params, samples)

    def loadFile(self, fpath):
        coco = {}

        for split in ['train', 'valid', 'test']:
            list_sent = []
            list_img_feat = []
            if sys.version_info < (3, 0):
                with open(os.path.join(fpath, split + '.pkl')) as f:
                    cocodata = pickle.load(f)
            else:
                with open(os.path.join(fpath, split + '.pkl'), 'rb') as f:
                    cocodata = pickle.load(f, encoding='latin1')

            for imgkey in range(len(cocodata['features'])):
                assert len(cocodata['image_to_caption_ids'][imgkey]) >= 5, \
                       cocodata['image_to_caption_ids'][imgkey]
                for captkey in cocodata['image_to_caption_ids'][imgkey][0:5]:
                    sent = cocodata['captions'][captkey]['cleaned_caption']
                    sent += ' .'  # add punctuation to end of sentence in COCO
                    list_sent.append(sent.encode('utf-8').split())
                    list_img_feat.append(cocodata['features'][imgkey])
            assert len(list_sent) == len(list_img_feat) and \
                len(list_sent) % 5 == 0
            list_img_feat = np.array(list_img_feat).astype('float32')
            coco[split] = {'sent': list_sent, 'imgfeat': list_img_feat}
        return coco['train'], coco['valid'], coco['test']

    def run(self, params, batcher):
        coco_embed = {'train': {'sentfeat': [], 'imgfeat': []},
                      'dev': {'sentfeat': [], 'imgfeat': []},
                      'test': {'sentfeat': [], 'imgfeat': []}}

        for key in self.coco_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            self.coco_data[key]['sent'] = np.array(self.coco_data[key]['sent'])
            self.coco_data[key]['sent'], idx_sort = np.sort(self.coco_data[key]['sent']), np.argsort(self.coco_data[key]['sent'])
            idx_unsort = np.argsort(idx_sort)

            coco_embed[key]['X'] = []
            nsent = len(self.coco_data[key]['sent'])
            for ii in range(0, nsent, params.batch_size):
                batch = self.coco_data[key]['sent'][ii:ii + params.batch_size]
                embeddings = batcher(params, batch)
                coco_embed[key]['sentfeat'].append(embeddings)
            coco_embed[key]['sentfeat'] = np.vstack(coco_embed[key]['sentfeat'])[idx_unsort]
            coco_embed[key]['imgfeat'] = np.array(self.coco_data[key]['imgfeat'])
            logging.info('Computed {0} embeddings'.format(key))

        config = {'seed': self.seed, 'projdim': 1000, 'margin': 0.2}
        clf = ImageSentenceRankingPytorch(train=coco_embed['train'],
                                          valid=coco_embed['dev'],
                                          test=coco_embed['test'],
                                          config=config)

        bestdevscore, r1_i2t, r5_i2t, r10_i2t, medr_i2t, \
            r1_t2i, r5_t2i, r10_t2i, medr_t2i = clf.run()

        logging.debug("\nTest scores | Image to text: \
            {0}, {1}, {2}, {3}".format(r1_i2t, r5_i2t, r10_i2t, medr_i2t))
        logging.debug("Test scores | Text to image: \
            {0}, {1}, {2}, {3}\n".format(r1_t2i, r5_t2i, r10_t2i, medr_t2i))

        return {'devacc': bestdevscore,
                'acc': [(r1_i2t, r5_i2t, r10_i2t, medr_i2t),
                        (r1_t2i, r5_t2i, r10_t2i, medr_t2i)],
                'ndev': len(coco_embed['dev']['sentfeat']),
                'ntest': len(coco_embed['test']['sentfeat'])}
