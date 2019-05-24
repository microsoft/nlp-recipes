"""This script reuses some code from
https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples
/run_classifier.py"""

import numpy as np
from tqdm import tqdm, trange

import torch

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertForTokenClassification

from common_ner import Language, create_data_loader, get_device


class BertTokenClassifier:
    """BERT-based token classifier."""

    def __init__(
        self,
        language=Language.ENGLISH,
        num_labels=2,
        cache_dir=".",
    ):

        """
        Initializes the classifier and the underlying pre-trained model.

        Args:
            language (Language, optional): The pre-trained model's language.
                The value of this argument determines which BERT model is
                used:
                    Language.ENGLISH: "bert-base-uncased"
                    Language.ENGLISHCASED: "bert-base-cased"
                    Language.ENGLISHLARGE: "bert-large-uncased"
                    Language.ENGLISHLARGECASED: "bert-large-cased"
                    Language.CHINESE: "bert-base-chinese"
                    Language.MULTILINGUAL: "bert-base-multilingual-cased"
                Defaults to Language.ENGLISH.
            num_labels (int, optional): The number of unique labels in the
                data. Defaults to 2.
            cache_dir (str, optional): Location of BERT's cache directory.
                Defaults to ".".
        """

        if num_labels < 2:
            raise Exception("Number of labels should be at least 2.")

        self.language = language
        self.num_labels = num_labels
        self.cache_dir = cache_dir

        self._model = BertForTokenClassification.from_pretrained(
            language.value,
            cache_dir=cache_dir,
            num_labels=num_labels,
        )

    @property
    def model(self):
        return self._model

    def _get_optimizer(self, learning_rate):
        """
        Initializes the optimizer and configure parameters to apply weight
        decay on.
        """
        param_optimizer = list(self.model.named_parameters())
        no_decay_params = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        params_weight_decay = 0.01
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay_params)
                ],
                "weight_decay": params_weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay_params)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = BertAdam(
            optimizer_grouped_parameters, lr=learning_rate
        )

        return optimizer

    def fit(self, token_ids, input_mask, labels,
            device="gpu", use_multiple_gpus=True,
            num_epochs=2, batch_size=32, learning_rate=5e-5,
            clip_gradient=False,
            max_gradient_norm=1.0):
        """
        Fine-tunes the BERT classifier using the given training data.

        Args:
            token_ids (list): List of lists. Each sublist contains
                numerical token ids corresponding tokens in the input text
                data.
            input_mask (list): List of lists. Each sublist contains attention
                masks of the input token id lists. 1 for input tokens and 0
                for padded tokens, so that padded tokens are not attended to.
            labels (list): List of lists, each sublist contains numerical
                token labels of a input sentence/paragraph.
            device (str, optional): Device used for training, "cpu" or
                "gpu". Default value is "gpu".
            use_multiple_gpus (bool, optional): Whether to use multiple GPUs
                if available. Default value is True.
            num_epochs (int, optional): Number of training epochs.
                Defaults to 1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            learning_rate (float, optional): learning rate of the BertAdam
                optimizer.
            clip_gradient (bool, optional): Whether to perform gradient
                clipping. Default value is False.
            max_gradient_norm (float, optional): Maximum gradient norm to
                apply gradient clipping on. Default value is 1.0.
        """
        train_dataloader = create_data_loader(input_ids=token_ids,
                                              input_mask=input_mask,
                                              label_ids=labels,
                                              sample_method='random',
                                              batch_size=batch_size)

        device = get_device(device)
        self.model.to(device)

        n_gpus = torch.cuda.device_count()
        if use_multiple_gpus and n_gpus > 1:
            self._model = torch.nn.DataParallel(self.model)

        optimizer = self._get_optimizer(learning_rate=learning_rate)

        self.model.train()
        for _ in trange(int(num_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            for step, batch in enumerate(
                tqdm(train_dataloader, desc="Iteration", mininterval=30)
            ):
                batch = tuple(t.to(device) for t in batch)
                b_token_ids, b_input_mask, b_label_ids = batch

                loss = self.model(
                    input_ids=b_token_ids,
                    attention_mask=b_input_mask,
                    labels=b_label_ids,
                )

                if n_gpus > 1:
                    # mean() to average on multi-gpu.
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_steps += 1

                ## TODO: compare with and without clip gradient
                if clip_gradient:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        max_norm=max_gradient_norm,
                    )

                optimizer.step()
                optimizer.zero_grad()

            train_loss = tr_loss / nb_tr_steps
            print("Train loss: {}".format(train_loss))

    def predict(self, token_ids, input_mask,
                labels=None, batch_size=32,
                device="gpu"):
        """
        Predict token labels on the testing data.

        Args:
            token_ids (list): List of lists. Each sublist contains
                numerical token ids corresponding tokens in the input text
                data.
            input_mask (list): List of lists. Each sublist contains attention
                masks of the input token id lists. 1 for input tokens and 0
                for padded tokens, so that padded tokens are not attended to.
            labels (list, optional): List of lists, each sublist contains
                numerical token labels of a input sentence/paragraph.
                If provided, it's used to compute the evaluation loss.
                Default value is None.
            batch_size (int, optional): Training batch size. Defaults to 32.
            device (str, optional): Device used for training, "cpu" or
                "gpu". Default value is "gpu".

        Returns:
            list: List of lists of predicted token labels.
        """
        test_dataloader = create_data_loader(input_ids=token_ids,
                                             input_mask=input_mask,
                                             label_ids=labels,
                                             batch_size=batch_size,
                                             sample_method="sequential")

        device = get_device(device)
        self.model.to(device)

        self.model.eval()
        predictions = []
        true_labels = []
        eval_loss = 0
        nb_eval_steps = 0
        for step, batch in enumerate(
            tqdm(test_dataloader, desc="Iteration", mininterval=10)
        ):
            batch = tuple(t.to(device) for t in batch)
            true_label_available = False
            if labels:
                b_input_ids, b_input_mask, b_labels = batch
                true_label_available = True
            else:
                b_input_ids, b_input_mask = batch

            with torch.no_grad():
                if true_label_available:
                    tmp_eval_loss = self.model(
                        b_input_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                    )
                logits = self.model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                )

            logits = logits.detach().cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

            if true_label_available:
                label_ids = b_labels.to("cpu").numpy()
                true_labels.append(label_ids)
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

        validation_loss = eval_loss / nb_eval_steps
        print("Evaluation loss: {}".format(validation_loss))

        return predictions


def postprocess_token_labels(labels, input_mask, label_map=None):
    """
    Removes predictions on padded tokens and maps predicted numerical labels
    back to original labels if label_map is provided.

    Args:
        labels (list): List of lists of predicted token labels.
        input_mask (list): List of lists. Each sublist contains attention
            masks of the input token id lists. 1 for input tokens and 0
            for padded tokens.
        label_map (dict, optional): A dictionary mapping original labels
            (which may be string type) to numerical label ids. If
            provided, it's used to map predicted numerical labels back to
            original labels. Default value is None.
    """
    if label_map:
        reversed_label_map = {v: k for k, v in label_map.items()}
        labels_org = [
            [reversed_label_map[l_i] for l_i in l]
            for l in labels
        ]

    else:
        labels_org = labels

    labels_org_no_padding = []
    for l, m in zip(labels_org, input_mask):
        l_np = np.array(l)
        m_np = np.array(m)
        l_no_padding = list(l_np[np.where(m_np == 1)])

        labels_org_no_padding.append(l_no_padding)

    return labels_org_no_padding
