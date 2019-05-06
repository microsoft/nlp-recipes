# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Image Annotation/Search for COCO with Pytorch
"""
from __future__ import absolute_import, division, unicode_literals

import logging
import copy
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim


class COCOProjNet(nn.Module):
    def __init__(self, config):
        super(COCOProjNet, self).__init__()
        self.imgdim = config['imgdim']
        self.sentdim = config['sentdim']
        self.projdim = config['projdim']
        self.imgproj = nn.Sequential(
                        nn.Linear(self.imgdim, self.projdim),
                        )
        self.sentproj = nn.Sequential(
                        nn.Linear(self.sentdim, self.projdim),
                        )

    def forward(self, img, sent, imgc, sentc):
        # imgc : (bsize, ncontrast, imgdim)
        # sentc : (bsize, ncontrast, sentdim)
        # img : (bsize, imgdim)
        # sent : (bsize, sentdim)
        img = img.unsqueeze(1).expand_as(imgc).contiguous()
        img = img.view(-1, self.imgdim)
        imgc = imgc.view(-1, self.imgdim)
        sent = sent.unsqueeze(1).expand_as(sentc).contiguous()
        sent = sent.view(-1, self.sentdim)
        sentc = sentc.view(-1, self.sentdim)

        imgproj = self.imgproj(img)
        imgproj = imgproj / torch.sqrt(torch.pow(imgproj, 2).sum(1, keepdim=True)).expand_as(imgproj)
        imgcproj = self.imgproj(imgc)
        imgcproj = imgcproj / torch.sqrt(torch.pow(imgcproj, 2).sum(1, keepdim=True)).expand_as(imgcproj)
        sentproj = self.sentproj(sent)
        sentproj = sentproj / torch.sqrt(torch.pow(sentproj, 2).sum(1, keepdim=True)).expand_as(sentproj)
        sentcproj = self.sentproj(sentc)
        sentcproj = sentcproj / torch.sqrt(torch.pow(sentcproj, 2).sum(1, keepdim=True)).expand_as(sentcproj)
        # (bsize*ncontrast, projdim)

        anchor1 = torch.sum((imgproj*sentproj), 1)
        anchor2 = torch.sum((sentproj*imgproj), 1)
        img_sentc = torch.sum((imgproj*sentcproj), 1)
        sent_imgc = torch.sum((sentproj*imgcproj), 1)

        # (bsize*ncontrast)
        return anchor1, anchor2, img_sentc, sent_imgc

    def proj_sentence(self, sent):
        output = self.sentproj(sent)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output # (bsize, projdim)

    def proj_image(self, img):
        output = self.imgproj(img)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output # (bsize, projdim)


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss
    """
    def __init__(self, margin):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor1, anchor2, img_sentc, sent_imgc):

        cost_sent = torch.clamp(self.margin - anchor1 + img_sentc,
                                min=0.0).sum()
        cost_img = torch.clamp(self.margin - anchor2 + sent_imgc,
                               min=0.0).sum()
        loss = cost_sent + cost_img
        return loss


class ImageSentenceRankingPytorch(object):
    # Image Sentence Ranking on COCO with Pytorch
    def __init__(self, train, valid, test, config):
        # fix seed
        self.seed = config['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.train = train
        self.valid = valid
        self.test = test

        self.imgdim = len(train['imgfeat'][0])
        self.sentdim = len(train['sentfeat'][0])
        self.projdim = config['projdim']
        self.margin = config['margin']

        self.batch_size = 128
        self.ncontrast = 30
        self.maxepoch = 20
        self.early_stop = True

        config_model = {'imgdim': self.imgdim,'sentdim': self.sentdim,
                        'projdim': self.projdim}
        self.model = COCOProjNet(config_model).cuda()

        self.loss_fn = PairwiseRankingLoss(margin=self.margin).cuda()

        self.optimizer = optim.Adam(self.model.parameters())

    def prepare_data(self, trainTxt, trainImg, devTxt, devImg,
                     testTxt, testImg):
        trainTxt = torch.FloatTensor(trainTxt)
        trainImg = torch.FloatTensor(trainImg)
        devTxt = torch.FloatTensor(devTxt).cuda()
        devImg = torch.FloatTensor(devImg).cuda()
        testTxt = torch.FloatTensor(testTxt).cuda()
        testImg = torch.FloatTensor(testImg).cuda()

        return trainTxt, trainImg, devTxt, devImg, testTxt, testImg

    def run(self):
        self.nepoch = 0
        bestdevscore = -1
        early_stop_count = 0
        stop_train = False

        # Preparing data
        logging.info('prepare data')
        trainTxt, trainImg, devTxt, devImg, testTxt, testImg = \
            self.prepare_data(self.train['sentfeat'], self.train['imgfeat'],
                              self.valid['sentfeat'], self.valid['imgfeat'],
                              self.test['sentfeat'], self.test['imgfeat'])

        # Training
        while not stop_train and self.nepoch <= self.maxepoch:
            logging.info('start epoch')
            self.trainepoch(trainTxt, trainImg, devTxt, devImg, nepoches=1)
            logging.info('Epoch {0} finished'.format(self.nepoch))

            results = {'i2t': {'r1': 0, 'r5': 0, 'r10': 0, 'medr': 0},
                       't2i': {'r1': 0, 'r5': 0, 'r10': 0, 'medr': 0},
                       'dev': bestdevscore}
            score = 0
            for i in range(5):
                devTxt_i = devTxt[i*5000:(i+1)*5000]
                devImg_i = devImg[i*5000:(i+1)*5000]
                # Compute dev ranks img2txt
                r1_i2t, r5_i2t, r10_i2t, medr_i2t = self.i2t(devImg_i,
                                                             devTxt_i)
                results['i2t']['r1'] += r1_i2t / 5
                results['i2t']['r5'] += r5_i2t / 5
                results['i2t']['r10'] += r10_i2t / 5
                results['i2t']['medr'] += medr_i2t / 5
                logging.info("Image to text: {0}, {1}, {2}, {3}"
                             .format(r1_i2t, r5_i2t, r10_i2t, medr_i2t))
                # Compute dev ranks txt2img
                r1_t2i, r5_t2i, r10_t2i, medr_t2i = self.t2i(devImg_i,
                                                             devTxt_i)
                results['t2i']['r1'] += r1_t2i / 5
                results['t2i']['r5'] += r5_t2i / 5
                results['t2i']['r10'] += r10_t2i / 5
                results['t2i']['medr'] += medr_t2i / 5
                logging.info("Text to Image: {0}, {1}, {2}, {3}"
                             .format(r1_t2i, r5_t2i, r10_t2i, medr_t2i))
                score += (r1_i2t + r5_i2t + r10_i2t +
                          r1_t2i + r5_t2i + r10_t2i) / 5

            logging.info("Dev mean Text to Image: {0}, {1}, {2}, {3}".format(
                        results['t2i']['r1'], results['t2i']['r5'],
                        results['t2i']['r10'], results['t2i']['medr']))
            logging.info("Dev mean Image to text: {0}, {1}, {2}, {3}".format(
                        results['i2t']['r1'], results['i2t']['r5'],
                        results['i2t']['r10'], results['i2t']['medr']))

            # early stop on Pearson
            if score > bestdevscore:
                bestdevscore = score
                bestmodel = copy.deepcopy(self.model)
            elif self.early_stop:
                if early_stop_count >= 3:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel

        # Compute test for the 5 splits
        results = {'i2t': {'r1': 0, 'r5': 0, 'r10': 0, 'medr': 0},
                   't2i': {'r1': 0, 'r5': 0, 'r10': 0, 'medr': 0},
                   'dev': bestdevscore}
        for i in range(5):
            testTxt_i = testTxt[i*5000:(i+1)*5000]
            testImg_i = testImg[i*5000:(i+1)*5000]
            # Compute test ranks img2txt
            r1_i2t, r5_i2t, r10_i2t, medr_i2t = self.i2t(testImg_i, testTxt_i)
            results['i2t']['r1'] += r1_i2t / 5
            results['i2t']['r5'] += r5_i2t / 5
            results['i2t']['r10'] += r10_i2t / 5
            results['i2t']['medr'] += medr_i2t / 5
            # Compute test ranks txt2img
            r1_t2i, r5_t2i, r10_t2i, medr_t2i = self.t2i(testImg_i, testTxt_i)
            results['t2i']['r1'] += r1_t2i / 5
            results['t2i']['r5'] += r5_t2i / 5
            results['t2i']['r10'] += r10_t2i / 5
            results['t2i']['medr'] += medr_t2i / 5

        return bestdevscore, results['i2t']['r1'], results['i2t']['r5'], \
                             results['i2t']['r10'], results['i2t']['medr'], \
                             results['t2i']['r1'], results['t2i']['r5'], \
                             results['t2i']['r10'], results['t2i']['medr']

    def trainepoch(self, trainTxt, trainImg, devTxt, devImg, nepoches=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + nepoches):
            permutation = list(np.random.permutation(len(trainTxt)))
            all_costs = []
            for i in range(0, len(trainTxt), self.batch_size):
                # forward
                if i % (self.batch_size*500) == 0 and i > 0:
                    logging.info('samples : {0}'.format(i))
                    r1_i2t, r5_i2t, r10_i2t, medr_i2t = self.i2t(devImg,
                                                                 devTxt)
                    logging.info("Image to text: {0}, {1}, {2}, {3}".format(
                        r1_i2t, r5_i2t, r10_i2t, medr_i2t))
                    # Compute test ranks txt2img
                    r1_t2i, r5_t2i, r10_t2i, medr_t2i = self.t2i(devImg,
                                                                 devTxt)
                    logging.info("Text to Image: {0}, {1}, {2}, {3}".format(
                        r1_t2i, r5_t2i, r10_t2i, medr_t2i))
                idx = torch.LongTensor(permutation[i:i + self.batch_size])
                imgbatch = Variable(trainImg.index_select(0, idx)).cuda()
                sentbatch = Variable(trainTxt.index_select(0, idx)).cuda()

                idximgc = np.random.choice(permutation[:i] +
                                           permutation[i + self.batch_size:],
                                           self.ncontrast*idx.size(0))
                idxsentc = np.random.choice(permutation[:i] +
                                            permutation[i + self.batch_size:],
                                            self.ncontrast*idx.size(0))
                idximgc = torch.LongTensor(idximgc)
                idxsentc = torch.LongTensor(idxsentc)
                # Get indexes for contrastive images and sentences
                imgcbatch = Variable(trainImg.index_select(0, idximgc)).view(
                    -1, self.ncontrast, self.imgdim).cuda()
                sentcbatch = Variable(trainTxt.index_select(0, idxsentc)).view(
                    -1, self.ncontrast, self.sentdim).cuda()

                anchor1, anchor2, img_sentc, sent_imgc = self.model(
                    imgbatch, sentbatch, imgcbatch, sentcbatch)
                # loss
                loss = self.loss_fn(anchor1, anchor2, img_sentc, sent_imgc)
                all_costs.append(loss.data.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += nepoches

    def t2i(self, images, captions):
        """
        Images: (5N, imgdim) matrix of images
        Captions: (5N, sentdim) matrix of captions
        """
        with torch.no_grad():
            # Project images and captions
            img_embed, sent_embed = [], []
            for i in range(0, len(images), self.batch_size):
                img_embed.append(self.model.proj_image(
                    Variable(images[i:i + self.batch_size])))
                sent_embed.append(self.model.proj_sentence(
                    Variable(captions[i:i + self.batch_size])))
            img_embed = torch.cat(img_embed, 0).data
            sent_embed = torch.cat(sent_embed, 0).data

            npts = int(img_embed.size(0) / 5)
            idxs = torch.cuda.LongTensor(range(0, len(img_embed), 5))
            ims = img_embed.index_select(0, idxs)

            ranks = np.zeros(5 * npts)
            for index in range(npts):

                # Get query captions
                queries = sent_embed[5*index: 5*index + 5]

                # Compute scores
                scores = torch.mm(queries, ims.transpose(0, 1)).cpu().numpy()
                inds = np.zeros(scores.shape)
                for i in range(len(inds)):
                    inds[i] = np.argsort(scores[i])[::-1]
                    ranks[5 * index + i] = np.where(inds[i] == index)[0][0]

            # Compute metrics
            r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
            r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
            r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
            medr = np.floor(np.median(ranks)) + 1
            return (r1, r5, r10, medr)

    def i2t(self, images, captions):
        """
        Images: (5N, imgdim) matrix of images
        Captions: (5N, sentdim) matrix of captions
        """
        with torch.no_grad():
            # Project images and captions
            img_embed, sent_embed = [], []
            for i in range(0, len(images), self.batch_size):
                img_embed.append(self.model.proj_image(
                    Variable(images[i:i + self.batch_size])))
                sent_embed.append(self.model.proj_sentence(
                    Variable(captions[i:i + self.batch_size])))
            img_embed = torch.cat(img_embed, 0).data
            sent_embed = torch.cat(sent_embed, 0).data

            npts = int(img_embed.size(0) / 5)
            index_list = []

            ranks = np.zeros(npts)
            for index in range(npts):

                # Get query image
                query_img = img_embed[5 * index]

                # Compute scores
                scores = torch.mm(query_img.view(1, -1),
                                  sent_embed.transpose(0, 1)).view(-1)
                scores = scores.cpu().numpy()
                inds = np.argsort(scores)[::-1]
                index_list.append(inds[0])

                # Score
                rank = 1e20
                for i in range(5*index, 5*index + 5, 1):
                    tmp = np.where(inds == i)[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks[index] = rank

            # Compute metrics
            r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
            r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
            r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
            medr = np.floor(np.median(ranks)) + 1
            return (r1, r5, r10, medr)
