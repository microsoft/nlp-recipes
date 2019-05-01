#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run script."""
import numpy as np
import logging
import argparse
import os
import sys

sys.path.append('.')  # Required to run on the MILA machines with SLURM

import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from utils import BufferedDataIterator, NLIIterator, compute_validation_loss
from models import MultitaskModel

from azureml.core.run import Run

# parser = argparse.ArgumentParser()
# parser.add_argument('--data_folder', type=str, help='data folder')
#
# args = parser.parse_args()
#
# input_data = args.input_data
# import os
# os.chdir(input_data)
# print("the input data is at %s" % input_data)
#
# get the Azure ML run object
run = Run.get_context()


cudnn.benchmark = True


def read_config(json_file):
    """Read JSON config."""
    json_object = json.load(open(json_file, 'r', encoding='utf-8'))
    return json_object

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
parser.add_argument('--data_folder', type=str, help='data folder')
# Add learning rate to tune model.
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
args = parser.parse_args()
data_folder = args.data_folder
#os.chdir(data_folder)

config_file_path = args.config
config = read_config(config_file_path)
save_dir = config['data']['save_dir']
load_dir = config['data']['load_dir']

os.chdir(data_folder)

if not os.path.exists('./log'):
    os.mkdir('./log')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (config['data']['task']),
    filemode='w'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

batch_size = config['training']['batch_size']
src_vocab_size = config['model']['n_words_src']
trg_vocab_size = config['model']['n_words_trg']
max_len_src = config['data']['max_src_length']
max_len_trg = config['data']['max_trg_length']

train_src = [item['train_src'] for item in config['data']['paths']]
train_trg = [item['train_trg'] for item in config['data']['paths']]
val_src = [item['val_src'] for item in config['data']['paths']]
val_trg = [item['val_trg'] for item in config['data']['paths']]
tasknames = [item['taskname'] for item in config['data']['paths']]

# Keep track of indicies to train forward and backward jointly
if 'skipthought_next' in tasknames and 'skipthought_previous' in tasknames:
    skipthought_idx = tasknames.index('skipthought_next')
    skipthought_backward_idx = tasknames.index('skipthought_previous')
    paired_tasks = {
        skipthought_idx: skipthought_backward_idx,
        skipthought_backward_idx: skipthought_idx
    }
else:
    paired_tasks = None
    skipthought_idx = None
    skipthought_backward_idx = None

train_iterator = BufferedDataIterator(
    train_src, train_trg,
    src_vocab_size, trg_vocab_size,
    tasknames, save_dir, buffer_size=1e6, lowercase=True
)

nli_iterator = NLIIterator(
    train=config['data']['nli_train'],
    dev=config['data']['nli_dev'],
    test=config['data']['nli_test'],
    vocab_size=-1,
    vocab=os.path.join(save_dir, 'src_vocab.pkl')
)

src_vocab_size = len(train_iterator.src[0]['word2id'])
trg_vocab_size = len(train_iterator.trg[0]['word2id'])

logging.info('Finished creating iterator ...')
logging.info('Found %d words in source : ' % (
    len(train_iterator.src[0]['id2word'])
))
for idx, taskname in enumerate(tasknames):
    logging.info('Found %d target words in task %s ' % (
        len(train_iterator.trg[idx]['id2word']), taskname)
    )

logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Source Word Embedding Dim  : %s' % (
    config['model']['dim_word_src'])
)
logging.info('Target Word Embedding Dim  : %s' % (
    config['model']['dim_word_trg'])
)
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim_src']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim_trg']))
logging.info('Source RNN Bidirectional  : %s' % (
    config['model']['bidirectional'])
)
logging.info('Batch Size : %d ' % (config['training']['batch_size']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

logging.info('Found %d words in src ' % (src_vocab_size))
logging.info('Found %d words in trg ' % (trg_vocab_size))

weight_mask = torch.ones(trg_vocab_size).cuda()
weight_mask[train_iterator.trg[0]['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()
nli_criterion = nn.CrossEntropyLoss().cuda()

model = MultitaskModel(
    src_emb_dim=config['model']['dim_word_src'],
    trg_emb_dim=config['model']['dim_word_trg'],
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_hidden_dim=config['model']['dim_src'],
    trg_hidden_dim=config['model']['dim_trg'],
    bidirectional=config['model']['bidirectional'],
    pad_token_src=train_iterator.src[0]['word2id']['<pad>'],
    pad_token_trg=train_iterator.trg[0]['word2id']['<pad>'],
    nlayers_src=config['model']['n_layers_src'],
    dropout=config['model']['dropout'],
    num_tasks=len(train_iterator.src),
    paired_tasks=paired_tasks
).cuda()

logging.info(model)

n_gpus = config['training']['n_gpus']
model = torch.nn.DataParallel(
    model, device_ids=range(n_gpus)
)

if load_dir == 'auto':
    ckpt = os.path.join(save_dir, 'best_model.model')
    if os.path.exists(ckpt):
        logging.info('Loading last saved model : %s ' % (ckpt))
        # Read file should be 'rb' cause it saves in 'wb'.
        model.load_state_dict(torch.load(open(ckpt, 'rb')))
    else:
        logging.info('Could not find model checkpoint, starting afresh')

elif load_dir and not load_dir == 'auto':
    logging.info('Loading model from specified checkpoint %s ' % (load_dir))
    model.load_state_dict(torch.load(
        open(load_dir, encoding='utf-8')
    ))

lr = args.learning_rate
# lr = config['training']['lrate']
optimizer = optim.Adam(model.parameters(), lr=lr)

task_losses = [[] for task in tasknames]
task_idxs = [0 for task in tasknames]
nli_losses = []
updates = 0
nli_ctr = 0
nli_epoch = 0
nli_mbatch_ctr = 0
mbatch_times = []
logging.info('Commencing Training ...')
rng_num_tasks = len(tasknames) - 1 if paired_tasks else len(tasknames)
while True:
    start = time.time()
    # Train NLI once every 10 minibatches of other tasks
    if nli_ctr % 10 == 0:
        minibatch = nli_iterator.get_parallel_minibatch(
            nli_mbatch_ctr, batch_size * n_gpus
        )
        optimizer.zero_grad()
        class_logits = model(
            minibatch, -1,
            return_hidden=False, paired_trg=None
        )

        loss = nli_criterion(
            class_logits.contiguous().view(-1, class_logits.size(1)),
            minibatch['labels'].contiguous().view(-1)
        )

        # nli_losses.append(loss.data[0])
        nli_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
        optimizer.step()

        # For AML.
        run.log('loss', loss.item())

        nli_mbatch_ctr += batch_size * n_gpus
        if nli_mbatch_ctr >= len(nli_iterator.train_lines):
            nli_mbatch_ctr = 0
            nli_epoch += 1
    else:
        # Sample a random task
        task_idx = np.random.randint(low=0, high=rng_num_tasks)

        # Get a minibatch corresponding to the sampled task
        minibatch = train_iterator.get_parallel_minibatch(
            task_idx, task_idxs[task_idx], batch_size * n_gpus,
            max_len_src, max_len_trg
        )

        '''Increment pointer into task and if current buffer is exhausted,
        fetch new buffer.'''
        task_idxs[task_idx] += batch_size * n_gpus
        if task_idxs[task_idx] >= train_iterator.buffer_size:
            train_iterator.fetch_buffer(task_idx)
            task_idxs[task_idx] = 0

        if task_idx == skipthought_idx:
            minibatch_back = train_iterator.get_parallel_minibatch(
                skipthought_backward_idx, task_idxs[skipthought_backward_idx],
                batch_size * n_gpus, max_len_src, max_len_trg
            )
            task_idxs[skipthought_backward_idx] += batch_size * n_gpus
            if (
                task_idxs[skipthought_backward_idx] >=
                train_iterator.buffer_size
            ):
                train_iterator.fetch_buffer(skipthought_backward_idx)
                task_idxs[skipthought_backward_idx] = 0

            optimizer.zero_grad()
            decoder_logit, decoder_logit_2 = model(
                minibatch, task_idx, paired_trg=minibatch_back['input_trg']
            )

            loss_f = loss_criterion(
                decoder_logit.contiguous().view(-1, decoder_logit.size(2)),
                minibatch['output_trg'].contiguous().view(-1)
            )

            loss_b = loss_criterion(
                decoder_logit_2.contiguous().view(-1, decoder_logit_2.size(2)),
                minibatch_back['output_trg'].contiguous().view(-1)
            )

            task_losses[task_idx].append(loss_f.data[0])
            task_losses[skipthought_backward_idx].append(loss_b.data[0])
            loss = loss_f + loss_b

        else:
            optimizer.zero_grad()
            decoder_logit = model(minibatch, task_idx)

            loss = loss_criterion(
                decoder_logit.contiguous().view(-1, decoder_logit.size(2)),
                minibatch['output_trg'].contiguous().view(-1)
            )

            task_losses[task_idx].append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
        optimizer.step()

    end = time.time()
    mbatch_times.append(end - start)

    if updates % config['management']['monitor_loss'] == 0 and updates != 0:
        for idx, task in enumerate(tasknames):
            logging.info('Seq2Seq Examples Processed : %d %s Loss : %.5f Num %s minibatches : %d' % (
                updates, task, np.mean(task_losses[idx]),
                task, len(task_losses[idx]))
            )
            run.log('Seq2Seq Examples Processed : ', updates)
            run.log('task: ', task)
            run.log('Loss: ', np.mean(task_losses[idx]))
            run.log('minibatches :', len(task_losses[idx]))

        logging.info('Round: %d NLI Epoch : %d NLI Examples Processed : %d NLI Loss : %.5f' % (
            nli_ctr, nli_epoch, nli_mbatch_ctr, np.mean(nli_losses)
        ))
        run.log('NLI Epoch : ', nli_epoch)
        run.log('NLI Examples Processed :', nli_mbatch_ctr)
        run.log('NLI Loss : ', np.mean(nli_losses))
        logging.info('Average time per mininbatch : %.5f' % (
            np.mean(mbatch_times)
        ))
        run.log('Average time per mininbatch : ', np.mean(mbatch_times))

        logging.info('******************************************************')
        task_losses = [[] for task in tasknames]
        mbatch_times = []
        nli_losses = []

    if updates % config['management']['checkpoint_freq'] == 0 and updates != 0:

        logging.info('Saving model ...')

        torch.save(
            model.state_dict(),
            open(os.path.join(
                save_dir,
                'best_model.model'), 'wb'
            )
        )
        # Let the training end.
        break

    if updates % config['management']['eval_freq'] == 0:
        logging.info('############################')
        logging.info('##### Evaluating model #####')
        logging.info('############################')
        for task_idx, task in enumerate(train_iterator.tasknames):
            if 'skipthought' in task:
                continue
            validation_loss = compute_validation_loss(
                config, model, train_iterator, loss_criterion,
                task_idx, lowercase=True
            )
            logging.info('%s Validation Loss : %.3f' % (
                task, validation_loss)
            )
        logging.info('Evaluating on NLI')
        n_correct = 0.
        n_wrong = 0.
        for j in range(0, len(nli_iterator.dev_lines), batch_size * n_gpus):
            minibatch = nli_iterator.get_parallel_minibatch(
                j, batch_size * n_gpus, 'dev'
            )
            class_logits = model(
                minibatch, -1,
                return_hidden=False, paired_trg=None
            )
            class_preds = F.softmax(class_logits).data.cpu().numpy().argmax(
                axis=-1
            )
            labels = minibatch['labels'].data.cpu().numpy()
            for pred, label in zip(class_preds, labels):
                if pred == label:
                    n_correct += 1.
                else:
                    n_wrong += 1.
        logging.info('NLI Dev Acc : %.5f' % (
            n_correct / (n_correct + n_wrong))
        )
        n_correct = 0.
        n_wrong = 0.
        for j in range(0, len(nli_iterator.test_lines), batch_size * n_gpus):
            minibatch = nli_iterator.get_parallel_minibatch(
                j, batch_size * n_gpus, 'test'
            )
            class_logits = model(
                minibatch, -1,
                return_hidden=False, paired_trg=None
            )
            class_preds = F.softmax(class_logits).data.cpu().numpy().argmax(
                axis=-1
            )
            labels = minibatch['labels'].data.cpu().numpy()
            for pred, label in zip(class_preds, labels):
                if pred == label:
                    n_correct += 1.
                else:
                    n_wrong += 1.
        logging.info('NLI Test Acc : %.5f' % (
            n_correct / (n_correct + n_wrong))
        )
        logging.info('******************************************************')

    updates += batch_size * n_gpus
    nli_ctr += 1