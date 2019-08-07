#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
The GenSen training process follows the steps:
1. Create or load the dataset vocabulary
2. Train on the training dataset for each batch epoch (batch size = 48 updates)
3. Evaluate on the validation dataset for every 10 epoches
4. Find the local minimum point on validation loss
5. Save the best model and stop the training process

AzureML provides AI Compute to train the model and track the performance.
This training process is based on GPU only.

"""
import argparse
import json
import logging
import os
import time

import horovod.torch as hvd
import mlflow
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from utils_nlp.models.gensen.multi_task_model import MultitaskModel
from utils_nlp.models.gensen.utils import (
    BufferedDataIterator,
    NLIIterator,
    compute_validation_loss,
)

cudnn.benchmark = True
logger = logging.getLogger(__name__)

hvd.init()
if torch.cuda.is_available():
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())


def metric_average(value, name):
    """
    Sync the validation loss with nodes.
    :param value:
    :param name:
    :return:
    """
    tensor = torch.tensor(value)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def setup_horovod(model, learning_rate):
    """ Setup for Horovod usage.

    Args:
        model(MultitaskModel): The MultitaskModel object.
        learning_rate(float): Learning rate for the model.

    Returns: hvd.DistributedOptimizer: Optimizer to use for computing
    gradients and applying updates.

    """
    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate * hvd.size())

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
    )

    return optimizer


def setup_logging(config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="log/%s" % (config["data"]["task"]),
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def log_config(config):
    logging.info("Model Parameters : ")
    logging.info("Task : %s " % (config["data"]["task"]))
    logging.info(
        "Source Word Embedding Dim  : %s" % (config["model"]["dim_word_src"])
    )
    logging.info(
        "Target Word Embedding Dim  : %s" % (config["model"]["dim_word_trg"])
    )
    logging.info("Source RNN Hidden Dim  : %s" % (config["model"]["dim_src"]))
    logging.info("Target RNN Hidden Dim  : %s" % (config["model"]["dim_trg"]))
    logging.info(
        "Source RNN Bidirectional  : %s" % (config["model"]["bidirectional"])
    )
    logging.info("Batch Size : %d " % (config["training"]["batch_size"]))
    logging.info("Optimizer : %s " % (config["training"]["optimizer"]))
    logging.info("Learning Rate : %f " % (config["training"]["lrate"]))


def evaluate(
    config,
    train_iterator,
    model,
    loss_criterion,
    monitor_epoch,
    min_val_loss,
    min_val_loss_epoch,
    save_dir,
    starting_time,
    model_state,
    max_epoch,
):
    """ Function to validate the model.

    Args:
        max_epoch(int): Limit training to specified number of epochs.
        model_state(dict): Saved model weights.
        config(dict): Config object.
        train_iterator(BufferedDataIterator): BufferedDataIterator object.
        model(MultitaskModel): The MultitaskModel object.
        loss_criterion(nn.CrossEntropyLoss): Cross entropy loss.
        monitor_epoch(int): Current epoch count.
        min_val_loss(float): Minimum validation loss
        min_val_loss_epoch(int): Epoch where the minimum validation
            loss was seen.
        save_dir(str): Directory path to save the model dictionary.
        starting_time(time.Time): Starting time of the training.

    Returns:
        bool: Whether to continue training or not.
    """

    break_flag = 0

    for task_idx, task in enumerate(train_iterator.tasknames):
        if "skipthought" in task:
            continue
        validation_loss = compute_validation_loss(
            config,
            model,
            train_iterator,
            loss_criterion,
            task_idx,
            lowercase=True,
        )
        validation_loss = metric_average(validation_loss, "val_loss")
        logging.info("%s Validation Loss : %.3f" % (task, validation_loss))

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            # log the best val accuracy to AML run
            logging.info(
                "Best Validation Loss: {}".format(np.float(validation_loss))
            )

        # If the validation loss is small enough, and it starts to go up.
        # Should stop training.
        # Small is defined by the number of epochs it lasts.
        if validation_loss < min_val_loss:
            min_val_loss = validation_loss
            min_val_loss_epoch = monitor_epoch
            model_state = model.state_dict()

        logging.info(
            "Monitor epoch: %d Validation Loss:  %.3f Min Validation Epoch: "
            "%d Loss : %.3f "
            % (
                monitor_epoch,
                validation_loss,
                min_val_loss_epoch,
                min_val_loss,
            )
        )
        if (monitor_epoch - min_val_loss_epoch) > config["training"][
            "stop_patience"
        ] or (max_epoch is not None and monitor_epoch >= max_epoch):
            logging.info("Saving model ...")
            # Save the name with validation loss.
            torch.save(
                model_state,
                open(os.path.join(save_dir, "best_model.model"), "wb"),
            )
            # Let the training end.
            break_flag = 1
            break
    if break_flag == 1:
        logging.info("##### Training stopped at ##### %f" % min_val_loss)
        logging.info(
            "##### Training Time ##### %f seconds"
            % (time.time() - starting_time)
        )
        return True, min_val_loss_epoch, min_val_loss, model_state
    else:
        return False, min_val_loss_epoch, min_val_loss, model_state


def evaluate_nli(nli_iterator, model, batch_size, n_gpus):
    """

    Args:
        nli_iterator(NLIIterator): NLIIterator object.
        model(MultitaskModel): Multitask model object.
        batch_size(int): Batch size.
        n_gpus(int): Number of gpus

    """
    n_correct = 0.0
    n_wrong = 0.0
    for j in range(0, len(nli_iterator.dev_lines), batch_size * n_gpus):
        minibatch = nli_iterator.get_parallel_minibatch(
            j, batch_size * n_gpus, "dev"
        )
        class_logits = model(
            minibatch, -1, return_hidden=False, paired_trg=None
        )
        class_preds = (
            f.softmax(class_logits).data.cpu().numpy().argmax(axis=-1)
        )
        labels = minibatch["labels"].data.cpu().numpy()
        for pred, label in zip(class_preds, labels):
            if pred == label:
                n_correct += 1.0
            else:
                n_wrong += 1.0
    logging.info("NLI Dev Acc : %.5f" % (n_correct / (n_correct + n_wrong)))
    n_correct = 0.0
    n_wrong = 0.0
    for j in range(0, len(nli_iterator.test_lines), batch_size * n_gpus):
        minibatch = nli_iterator.get_parallel_minibatch(
            j, batch_size * n_gpus, "test"
        )
        class_logits = model(
            minibatch, -1, return_hidden=False, paired_trg=None
        )
        class_preds = (
            f.softmax(class_logits).data.cpu().numpy().argmax(axis=-1)
        )
        labels = minibatch["labels"].data.cpu().numpy()
        for pred, label in zip(class_preds, labels):
            if pred == label:
                n_correct += 1.0
            else:
                n_wrong += 1.0
    logging.info("NLI Test Acc : %.5f" % (n_correct / (n_correct + n_wrong)))
    logging.info("******************************************************")


def train(config, data_folder, learning_rate=0.0001, max_epoch=None):
    """ Train the Gensen model.

    Args:
        max_epoch(int): Limit training to specified number of epochs.
        config(dict): Loaded json file as a python object.
        data_folder(str): Path to the folder containing the data.
        learning_rate(float): Learning rate for the model.
    """
    owd = os.getcwd()
    os.chdir(data_folder)

    try:
        with mlflow.start_run():
            save_dir = config["data"]["save_dir"]
            if not os.path.exists("./log"):
                os.makedirs("./log")

            os.makedirs(save_dir, exist_ok=True)

            setup_logging(config)

            batch_size = config["training"]["batch_size"]
            src_vocab_size = config["model"]["n_words_src"]
            trg_vocab_size = config["model"]["n_words_trg"]
            max_len_src = config["data"]["max_src_length"]
            max_len_trg = config["data"]["max_trg_length"]
            model_state = {}

            train_src = [item["train_src"] for item in config["data"]["paths"]]
            train_trg = [item["train_trg"] for item in config["data"]["paths"]]
            tasknames = [item["taskname"] for item in config["data"]["paths"]]

            # Keep track of indicies to train forward and backward jointly
            if (
                "skipthought_next" in tasknames
                and "skipthought_previous" in tasknames
            ):
                skipthought_idx = tasknames.index("skipthought_next")
                skipthought_backward_idx = tasknames.index(
                    "skipthought_previous"
                )
                paired_tasks = {
                    skipthought_idx: skipthought_backward_idx,
                    skipthought_backward_idx: skipthought_idx,
                }
            else:
                paired_tasks = None
                skipthought_idx = None
                skipthought_backward_idx = None

            train_iterator = BufferedDataIterator(
                train_src,
                train_trg,
                src_vocab_size,
                trg_vocab_size,
                tasknames,
                save_dir,
                buffer_size=1e6,
                lowercase=True,
                seed=(hvd.rank() + 1) * 12345,
            )

            nli_iterator = NLIIterator(
                train=config["data"]["nli_train"],
                dev=config["data"]["nli_dev"],
                test=config["data"]["nli_test"],
                vocab_size=-1,
                vocab=os.path.join(save_dir, "src_vocab.pkl"),
                seed=(hvd.rank() + 1) * 12345,
            )

            src_vocab_size = len(train_iterator.src[0]["word2id"])
            trg_vocab_size = len(train_iterator.trg[0]["word2id"])

            # Logging set up.
            logging.info("Finished creating iterator ...")
            log_config(config)
            logging.info(
                "Found %d words in source : "
                % (len(train_iterator.src[0]["id2word"]))
            )
            for idx, taskname in enumerate(tasknames):
                logging.info(
                    "Found %d target words in task %s "
                    % (len(train_iterator.trg[idx]["id2word"]), taskname)
                )
            logging.info("Found %d words in src " % src_vocab_size)
            logging.info("Found %d words in trg " % trg_vocab_size)

            weight_mask = torch.ones(trg_vocab_size).cuda()
            weight_mask[train_iterator.trg[0]["word2id"]["<pad>"]] = 0
            loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()
            nli_criterion = nn.CrossEntropyLoss().cuda()

            model = MultitaskModel(
                src_emb_dim=config["model"]["dim_word_src"],
                trg_emb_dim=config["model"]["dim_word_trg"],
                src_vocab_size=src_vocab_size,
                trg_vocab_size=trg_vocab_size,
                src_hidden_dim=config["model"]["dim_src"],
                trg_hidden_dim=config["model"]["dim_trg"],
                bidirectional=config["model"]["bidirectional"],
                pad_token_src=train_iterator.src[0]["word2id"]["<pad>"],
                pad_token_trg=train_iterator.trg[0]["word2id"]["<pad>"],
                nlayers_src=config["model"]["n_layers_src"],
                dropout=config["model"]["dropout"],
                num_tasks=len(train_iterator.src),
                paired_tasks=paired_tasks,
            ).cuda()

            optimizer = setup_horovod(model, learning_rate=learning_rate)
            logging.info(model)

            n_gpus = config["training"]["n_gpus"]
            model = torch.nn.DataParallel(model, device_ids=range(n_gpus))

            task_losses = [[] for _ in tasknames]
            task_idxs = [0 for _ in tasknames]
            nli_losses = []
            updates = 0
            nli_ctr = 0
            nli_epoch = 0
            monitor_epoch = 0
            nli_mbatch_ctr = 0
            mbatch_times = []
            min_val_loss = 10000000
            min_val_loss_epoch = -1
            rng_num_tasks = (
                len(tasknames) - 1 if paired_tasks else len(tasknames)
            )
            logging.info("OS Environ: \n {} \n\n".format(os.environ))
            mlflow.log_param("learning_rate", learning_rate)
            logging.info("Commencing Training ...")
            start = time.time()
            while True:
                batch_start_time = time.time()
                # Train NLI once every 10 minibatches of other tasks
                if nli_ctr % 10 == 0:
                    minibatch = nli_iterator.get_parallel_minibatch(
                        nli_mbatch_ctr, batch_size * n_gpus
                    )
                    optimizer.zero_grad()
                    class_logits = model(
                        minibatch, -1, return_hidden=False, paired_trg=None
                    )

                    loss = nli_criterion(
                        class_logits.contiguous().view(
                            -1, class_logits.size(1)
                        ),
                        minibatch["labels"].contiguous().view(-1),
                    )

                    # nli_losses.append(loss.data[0])
                    nli_losses.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                    optimizer.step()

                    nli_mbatch_ctr += batch_size * n_gpus
                    if nli_mbatch_ctr >= len(nli_iterator.train_lines):
                        nli_mbatch_ctr = 0
                        nli_epoch += 1
                else:
                    # Sample a random task
                    task_idx = np.random.randint(low=0, high=rng_num_tasks)

                    # Get a minibatch corresponding to the sampled task
                    minibatch = train_iterator.get_parallel_minibatch(
                        task_idx,
                        task_idxs[task_idx],
                        batch_size * n_gpus,
                        max_len_src,
                        max_len_trg,
                    )

                    """Increment pointer into task and if current buffer is
                    exhausted, fetch new buffer. """
                    task_idxs[task_idx] += batch_size * n_gpus
                    if task_idxs[task_idx] >= train_iterator.buffer_size:
                        train_iterator.fetch_buffer(task_idx)
                        task_idxs[task_idx] = 0

                    if task_idx == skipthought_idx:
                        minibatch_back = train_iterator.get_parallel_minibatch(
                            skipthought_backward_idx,
                            task_idxs[skipthought_backward_idx],
                            batch_size * n_gpus,
                            max_len_src,
                            max_len_trg,
                        )
                        task_idxs[skipthought_backward_idx] += (
                            batch_size * n_gpus
                        )
                        if (
                            task_idxs[skipthought_backward_idx]
                            >= train_iterator.buffer_size
                        ):
                            train_iterator.fetch_buffer(
                                skipthought_backward_idx
                            )
                            task_idxs[skipthought_backward_idx] = 0

                        optimizer.zero_grad()
                        decoder_logit, decoder_logit_2 = model(
                            minibatch,
                            task_idx,
                            paired_trg=minibatch_back["input_trg"],
                        )

                        loss_f = loss_criterion(
                            decoder_logit.contiguous().view(
                                -1, decoder_logit.size(2)
                            ),
                            minibatch["output_trg"].contiguous().view(-1),
                        )

                        loss_b = loss_criterion(
                            decoder_logit_2.contiguous().view(
                                -1, decoder_logit_2.size(2)
                            ),
                            minibatch_back["output_trg"].contiguous().view(-1),
                        )

                        task_losses[task_idx].append(loss_f.data[0])
                        task_losses[skipthought_backward_idx].append(
                            loss_b.data[0]
                        )
                        loss = loss_f + loss_b

                    else:
                        optimizer.zero_grad()
                        decoder_logit = model(minibatch, task_idx)

                        loss = loss_criterion(
                            decoder_logit.contiguous().view(
                                -1, decoder_logit.size(2)
                            ),
                            minibatch["output_trg"].contiguous().view(-1),
                        )

                        task_losses[task_idx].append(loss.item())

                    loss.backward()
                    # For distributed optimizer need to sync before gradient
                    # clipping.
                    optimizer.synchronize()

                    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                    optimizer.step()

                end = time.time()
                mbatch_times.append(end - batch_start_time)

                # Validations
                if (
                    updates % config["management"]["monitor_loss"] == 0
                    and updates != 0
                ):
                    monitor_epoch += 1
                    for idx, task in enumerate(tasknames):
                        logging.info(
                            "Seq2Seq Examples Processed : %d %s Loss : %.5f Num %s "
                            "minibatches : %d"
                            % (
                                updates,
                                task,
                                np.mean(task_losses[idx]),
                                task,
                                len(task_losses[idx]),
                            )
                        )
                        mlflow.log_metric(
                            "validation_loss",
                            np.mean(task_losses[idx]),
                            step=monitor_epoch,
                        )

                    logging.info(
                        "Round: %d NLI Epoch : %d NLI Examples Processed : %d NLI "
                        "Loss : %.5f "
                        % (
                            nli_ctr,
                            nli_epoch,
                            nli_mbatch_ctr,
                            np.mean(nli_losses),
                        )
                    )
                    mlflow.log_metric(
                        "nli_loss", np.mean(nli_losses), step=nli_epoch
                    )

                    logging.info(
                        "Average time per minibatch : %.5f"
                        % (np.mean(mbatch_times))
                    )
                    mlflow.log_metric(
                        "minibatch_avg_duration", np.mean(mbatch_times)
                    )

                    task_losses = [[] for _ in tasknames]
                    mbatch_times = []
                    nli_losses = []

                    # For validate and break if done.
                    logging.info("############################")
                    logging.info("##### Evaluating model #####")
                    logging.info("############################")
                    training_complete, min_val_loss_epoch, min_val_loss, model_state = evaluate(
                        config=config,
                        train_iterator=train_iterator,
                        model=model,
                        loss_criterion=loss_criterion,
                        monitor_epoch=monitor_epoch,
                        min_val_loss=min_val_loss,
                        min_val_loss_epoch=min_val_loss_epoch,
                        save_dir=save_dir,
                        starting_time=start,
                        model_state=model_state,
                        max_epoch=max_epoch,
                    )
                    if training_complete:
                        mlflow.log_metric("min_val_loss", float(min_val_loss))
                        mlflow.log_metric("learning_rate", learning_rate)
                        break

                    logging.info("Evaluating on NLI")
                    evaluate_nli(
                        nli_iterator=nli_iterator,
                        model=model,
                        n_gpus=n_gpus,
                        batch_size=batch_size,
                    )

                updates += batch_size * n_gpus
                nli_ctr += 1
                logging.info("Updates: %d" % updates)
    finally:
        os.chdir(owd)


def read_config(json_file):
    """Read JSON config."""
    json_object = json.load(open(json_file, "r", encoding="utf-8"))
    return json_object


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config", required=True)
    parser.add_argument("--data_folder", type=str, help="data folder")
    # Add learning rate to tune model.
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=None,
        help="Limit training to specified number of epochs.",
    )

    args = parser.parse_args()
    data_path = args.data_folder
    lr = args.learning_rate

    config_file_path = args.config
    max_epoch = args.max_epoch
    config_obj = read_config(config_file_path)
    train(config_obj, data_path, lr, max_epoch)
