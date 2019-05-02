import os

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE,\
    WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, \
    BertForTokenClassification, BertConfig
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import WordpieceTokenizer


import logging
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def get_device(local_rank, no_cuda):
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    return device, n_gpu


def load_model(model_config, path_config, device_config, global_config):
    cache_dir = path_config.cache_dir if path_config.cache_dir else \
        os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                     'distributed_{}'.format(device_config.local_rank))
    if model_config.model_type == 'sequence':
        model = BertForSequenceClassification.from_pretrained(
            model_config.bert_model, cache_dir=cache_dir,
            num_labels=model_config.num_labels)
    elif model_config.model_type == 'token':
        model = BertForTokenClassification.from_pretrained(
            model_config.bert_model, cache_dir=cache_dir,
            num_labels=model_config.num_labels)
    if global_config.fp16:
        model.half()
    model.to(device_config.device)
    if device_config.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif device_config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model


def configure_optimizer(optimizer_config, global_config, train_config,
                        device_config, model, num_train_examples):
    param_optimizer = list(model.named_parameters())
    no_decay = optimizer_config.no_decay_params
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    num_train_optimization_steps = int(num_train_examples /
                                       train_config.train_batch_size /
                                       train_config.gradient_accumulation_steps) * train_config.num_train_epochs
    if device_config.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    optimizer_config.num_train_optimization_steps \
        = num_train_optimization_steps

    if global_config.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=optimizer_config.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if optimizer_config.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale=optimizer_config.loss_scale)
        warmup_linear = WarmupLinearSchedule(
            warmup=optimizer_config.warmup_proportion,
            t_total=num_train_optimization_steps)

        return optimizer, optimizer_config, warmup_linear

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=optimizer_config.learning_rate,
                             warmup=optimizer_config.warmup_proportion,
                             t_total=num_train_optimization_steps)

        return optimizer, optimizer_config, None


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_sequence_features(examples,
                                          label_list,
                                          max_seq_length,
                                          output_mode,
                                          bert_model,
                                          do_lower_case):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def convert_examples_to_token_features(examples,
                                       label_list,
                                       max_seq_length,
                                       bert_model,
                                       do_lower_case):
    """Loads a data file into a list of `InputBatch`s."""
    tokenizer = BertTokenizer.from_pretrained(bert_model,
                                              do_lower_case=do_lower_case)
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        #         if ex_index % 10000 == 0:
        #             logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_lower = example.text_a.lower()
        new_labels = []
        tokens = []
        for word, tag in zip(text_lower.split(), example.label):
            # print('splitting: ', word)
            sub_words = tokenizer.wordpiece_tokenizer.tokenize(word)
            for count, sub_word in enumerate(sub_words):
                # print('subword: ',sub_word)
                if count > 0:
                    tag = 'X'
                new_labels.append(tag)
                tokens.append(sub_word)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            new_labels = new_labels[:max_seq_length]

        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        label_padding = ['O'] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        new_labels += label_padding

        if len(new_labels) != 75:
            print(len(new_labels))
            print(new_labels)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(new_labels) == max_seq_length

        label_id = [label_map[label] for label in new_labels]

        #         if ex_index < 5:
        #             logger.info("*** Example ***")
        #             logger.info("guid: %s" % (example.guid))
        #             logger.info("tokens: %s" % " ".join(
        #                     [str(x) for x in tokens]))
        #             logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #             logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #             logger.info(
        #                     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #             logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


##train_examples and label list can be contained within one data
# preprocessor object
def create_train_dataloader(train_examples, model_config,
                            train_config, label_list, device_config):
    if model_config.model_type == 'sequence':
        train_features = convert_examples_to_sequence_features(
            examples=train_examples,
            label_list=label_list,
            bert_model=model_config.bert_model,
            max_seq_length=model_config.max_seq_length,
            output_mode=model_config.output_mode,
            do_lower_case=model_config.do_lower_case)
    elif model_config.model_type == 'token':
        train_features = convert_examples_to_token_features(
            examples=train_examples,
            label_list=label_list,
            bert_model=model_config.bert_model,
            max_seq_length=model_config.max_seq_length,
            do_lower_case=model_config.do_lower_case)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    if model_config.output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    elif model_config.output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

    ## Create train data loader
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if device_config.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=train_config.train_batch_size)

    return train_dataloader


def create_eval_dataloader(eval_examples, model_config,
                           eval_config, label_list):
    if model_config.model_type == 'sequence':
        eval_features = convert_examples_to_sequence_features(
            examples=eval_examples,
            label_list=label_list,
            bert_model=model_config.bert_model,
            max_seq_length=model_config.max_seq_length,
            output_mode=model_config.output_mode,
            do_lower_case=model_config.do_lower_case)
    elif model_config.model_type == 'token':
        eval_features = convert_examples_to_token_features(
            examples=eval_examples,
            label_list=label_list,
            bert_model=model_config.bert_model,
            max_seq_length=model_config.max_seq_length,
            do_lower_case=model_config.do_lower_case)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if model_config.output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif model_config.output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=eval_config.eval_batch_size)

    return eval_dataloader, all_label_ids


def train_model(model, train_dataloader, optimizer,
                train_config, model_config, optimizer_config,
                device_config, global_config, warmup_linear=None):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    model.train()
    for _ in trange(int(train_config.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device_config.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None)

            if model_config.output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, model_config.num_labels),
                                label_ids.view(-1))
            elif model_config.output_mode == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if device_config.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if train_config.gradient_accumulation_steps > 1:
                loss = loss / train_config.gradient_accumulation_steps

            if global_config.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
                if global_config.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = optimizer_config.learning_rate * \
                                   warmup_linear.get_lr(
                                       global_step/optimizer_config.num_train_optimization_steps,
                                                                             optimizer_config.args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    loss = tr_loss/nb_tr_steps

    return model, loss


def save_model(model, tokenizer, path_config):
    model_to_save = model.module if hasattr(model,
                                            'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(path_config.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(path_config.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(path_config.output_dir)


def eval_model(model, eval_dataloader, model_config, device_config,
               eval_label_ids=None, eval_func=None):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device_config.device)
        input_mask = input_mask.to(device_config.device)
        segment_ids = segment_ids.to(device_config.device)
        label_ids = label_ids.to(device_config.device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        if model_config.output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, model_config.num_labels),
                                     label_ids.view(-1))
        elif model_config.output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if model_config.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif model_config.output_mode == "regression":
        preds = np.squeeze(preds)

    eval_result = {}
    if eval_label_ids and eval_func:
        eval_result['eval_metric'] = eval_func(preds, eval_label_ids.numpy())

        eval_result['eval_loss'] = eval_loss

        return preds, eval_result
    else:
        return preds, None

    # result['global_step'] = global_step
    # result['loss'] = loss
    #
    # output_eval_file = os.path.join(path_config.output_dir, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results *****")
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))


def train_token_model(model, train_dataloader, optimizer,
                      train_config, model_config, optimizer_config,
                      device_config, global_config, warmup_linear=None):
    for _ in trange(train_config.num_train_epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device_config.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=1)
            # update parameters performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule
            optimizer.step()
            model.zero_grad()#Zero the gradients before running the next batch.
        # print train loss per epoch
        train_loss = tr_loss/nb_tr_steps
        print("Train loss: {}".format(train_loss))
    return model, train_loss


def eval_token_model(model, eval_dataloader, model_config, device_config,
                     label_list, eval_func=None):
    from seqeval.metrics import f1_score

    model.eval()
    predictions = []
    true_labels = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in eval_dataloader:
        batch = tuple(t.to(device_config.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        label_ids = b_labels.to('cpu').numpy()
        true_labels.append(label_ids)
        tmp_eval_accuracy = eval_func(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    pred_tags = [[label_list[p_i] for p_i in p] for p in predictions]
    valid_tags = [[label_list[l_ii] for l_ii in l_i] for l in true_labels for
                  l_i in l]

    validation_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print("Validation loss: {}".format(validation_loss))
    print("Validation Accuracy: {}".format(eval_accuracy))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

    return pred_tags, validation_loss, eval_accuracy

