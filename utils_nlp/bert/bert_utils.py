"""This script reuses some code from
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py"""
from enum import Enum, auto

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.optim import Adam

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertForTokenClassification


def get_device():
    """
    Helper function for detecting devices and number of gpus.

    Returns:
        (str, int): tuple of device and number of gpus
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    return device, n_gpu


def create_token_feature_dataset(data,
                                 tokenizer,
                                 label_map,
                                 max_seq_length,
                                 true_label_available,
                                 trailing_piece_tag="X"):
    """
    Converts data from text to TensorDataset containing numerical features.

    Args:
        data (list): List of input data in BertInputData type, each contains
            three fields:
                text_a: text of the first sentence,
                text_b: text of the second sentence(optional)
                label: label (optional)
        tokenizer (BertTokenizer): Tokenizer for splitting sentence into
            word pieces.
        label_map (dict): Dictionary for mapping token labels to integers.
        max_seq_length (int): Maximum length of the list of tokens. Lists
            longer than this are truncated and shorter ones are padded with
            zeros.
        true_label_available (bool): Whether data labels are available.
        trailing_piece_tag (str): Tags used to label trailing word pieces.
            For example, "playing" is broken down into "play" and "##ing",
            "play" preserves its original label and "##ing" is labeled as "X".

    Returns:
        TensorDataset: A TensorDataset consisted of the following numerical
            feature tensors:
            1. token ids
            2. attention mask
            3. segment ids
            4. label ids, if true_label_available is True
    """

    features = []
    for ex_index, example in enumerate(data):

        text_lower = example.text_a.lower()
        new_labels = []
        tokens = []
        for word, tag in zip(text_lower.split(), example.label):
            # print('splitting: ', word)
            sub_words = tokenizer.wordpiece_tokenizer.tokenize(word)
            for count, sub_word in enumerate(sub_words):
                if tag is not None and count > 0:
                    tag = trailing_piece_tag
                new_labels.append(tag)
                tokens.append(sub_word)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            new_labels = new_labels[:max_seq_length]

        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1.0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0.0] * (max_seq_length - len(input_ids))
        label_padding = ["O"] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        new_labels += label_padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(new_labels) == max_seq_length

        label_ids = [label_map[label] for label in new_labels]

        features.append((input_ids, input_mask, segment_ids, label_ids))

    all_input_ids = torch.tensor([f[0] for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f[2] for f in features],
                                   dtype=torch.long)

    if true_label_available:
        all_label_ids = torch.tensor([f[3] for f in features],
                                     dtype=torch.long)
        tensor_data = TensorDataset(all_input_ids, all_input_mask,
                                    all_segment_ids, all_label_ids)
    else:
        tensor_data = TensorDataset(all_input_ids, all_input_mask,
                                    all_segment_ids)

    return tensor_data


class Language(Enum):
    """An enumeration of the supported languages."""

    ENGLISH = "bert-base-uncased"
    CHINESE = "bert-base-chinese"
    SPANISH = auto()
    HINDI = auto()
    FRENCH = auto()


class BertTokenClassifier:
    """BERT-based token classifier."""

    def __init__(self, config, label_map, device, n_gpu,
                 language=Language.ENGLISH, cache_dir="."):

        """
        Initializes the classifier and the underlying pretrained model and
        optimizer.

        Args:
            config (BERTFineTuneConfig): A configuration object contains
                settings of model, training, and optimizer.
            label_map (dict): Dictionary used to map token labels to
                integers during data preprocessing.
            device (str): "cuda" or "cpu". Can be obtained by calling
                get_device.
            n_gpu (int): Number of GPUs available.Can be obtained by calling
                get_device.
            language (Language, optinal): The pretrained model's language.
                Defaults to Language.ENGLISH.
            cache_dir (str, optional): Location of BERT's cache directory.
                Defaults to ".".
        """

        print("BERT fine tune configurations:")
        print(config)

        self.language = language
        self.device = device
        self.n_gpu = n_gpu
        self.num_labels = len(label_map)
        self.cache_dir = cache_dir
        self.label_map = label_map

        self.bert_model = config.bert_model

        self.optimizer_name = config.optimizer_name
        self.no_decay_params = config.no_decay_params
        self.params_weight_decay = config.params_weight_decay
        self.learning_rate = config.learning_rate
        self.clip_gradient = config.clip_gradient
        self.max_gradient_norm = config.max_gradient_norm

        self.num_train_epochs = config.num_train_epochs
        self.batch_size = config.batch_size

        # This step needs to be done before creating optimizer
        self.model = self._load_model()
        self.optimizer = self._get_optimizer()

        self._is_trained = False

    def _load_model(self):
        """Loads the pretrained BERT model."""
        model = BertForTokenClassification.from_pretrained(
            self.bert_model, cache_dir=self.cache_dir,
            num_labels=self.num_labels)

        model.to(self.device)

        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        return model

    def _get_optimizer(self):
        """
        Initializes the optimizer and configure parameters to apply weight
        decay on.
        """
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in self.no_decay_params)],
             'weight_decay': self.params_weight_decay},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in self.no_decay_params)],
             'weight_decay': 0.0}
        ]

        if self.optimizer_name == 'BertAdam':
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=self.learning_rate)
        elif self.optimizer_name == 'Adam':
            optimizer = Adam(optimizer_grouped_parameters,
                            lr=self.learning_rate)

        return optimizer

    def fit(self, train_dataset):
        """
        Fine-tunes the BERT classifier using the given training data.

        Args:
            train_dataset (TensorDataset): TensorDataset consisted of the
                following numerical feature tensors.
                1. token ids
                2. attention mask
                3. segment ids
                4. label ids
        """
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=self.batch_size)

        global_step = 0
        self.model.train()
        for _ in trange(int(self.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader,
                                              desc="Iteration",
                                              mininterval=30)):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = self.model(input_ids=input_ids,
                                  token_type_ids=segment_ids,
                                  attention_mask=input_mask,
                                  labels=label_ids)

                if self.n_gpu > 1:
                    # mean() to average on multi-gpu.
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_steps += 1

                if self.clip_gradient:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        max_norm=self.max_gradient_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1

            train_loss = tr_loss/nb_tr_steps
            print("Train loss: {}".format(train_loss))

        self._is_trained = True

    def predict(self, test_dataset):
        """
        Predict token labels on the testing data.

        Args:
            test_dataset (TensorDataset): TensorDataset consisted of the
                following numerical feature tensors.
                1. token ids
                2. attention mask
                3. segment ids
                4. label ids, optional

        Returns:
            tuple: The first element of the tuple is the predicted token
                labels. If the testing dataset contain label ids, the second
                element of the tuple is the true token labels.
        """
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                     batch_size=self.batch_size)

        if not self._is_trained:
            raise Exception("Model is not trained. Please train model before "
                            "predict.")

        self.model.eval()
        predictions = []
        true_labels = []
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps = 0
        for step, batch in enumerate(tqdm(test_dataloader,
                                          desc="Iteration",
                                          mininterval=10)):
            batch = tuple(t.to(self.device) for t in batch)
            true_label_available = False
            if len(batch) == 3:
                b_input_ids, b_input_mask, b_segment_ids = batch
            elif len(batch) == 4:
                b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
                true_label_available = True

            with torch.no_grad():
                if true_label_available:
                    tmp_eval_loss = self.model(b_input_ids,
                                               token_type_ids=None,
                                               attention_mask=b_input_mask,
                                               labels=b_labels)
                logits = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)

            logits = logits.detach().cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

            if true_label_available:
                label_ids = b_labels.to('cpu').numpy()
                true_labels.append(label_ids)
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

        validation_loss = eval_loss / nb_eval_steps
        print("Validation loss: {}".format(validation_loss))

        reversed_label_map = {v: k for k, v in self.label_map.items()}
        pred_tags = [[reversed_label_map[p_i] for p_i in p] for p in
                     predictions]

        if true_label_available:
            valid_tags = [[reversed_label_map[l_ii] for l_ii in l_i] for
                          l in true_labels for l_i in l]

            return pred_tags, valid_tags
        else:
            return pred_tags,



