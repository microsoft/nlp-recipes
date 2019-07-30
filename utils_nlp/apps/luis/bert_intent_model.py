# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import codecs
import json
import logging
import os
import sys

import pandas as pd
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report

from utils_nlp.models.bert.sequence_classification import (
    BERTSequenceClassifier,
)
from utils_nlp.models.bert.common import Language, Tokenizer
from utils_nlp.common.timer import Timer


logger = logging.getLogger(__name__)


class BERTIntentClassifier:
    """intent classifier which trains on luis model file """

    def __init__(
        self,
        language=Language.ENGLISH,
        to_lower=False,
        max_seq_length=100,
        num_gpus=None,
        num_epochs=5,
        batch_size=16,
        train_size=0.8,
        learning_rate=3e-5,
        warmup_proportion=None,
        cache_dir="./temp",
    ):
        """Initialize the classifier

        Args:
            language (Language, optional): The pretrained model's language.
                efaults to Language.ENGLISH.
            to_lower (boolean, optional):  Whether to lower case the input
            max_seq_length (int, optional): the maximum length for input text data 
                in training and prediction 
            num_gpus (int, optional): The number of gpus to use.
                If None is specified, all available GPUs will be used. Defaults to None.
            num_epochs (int, optional): Number of training epochs.
                Defaults to 1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            train_size (float, optional): portion of the input training data used for training,
                use 1.0 if all data needs for training.
            learning (float): Learning rate of the Adam optimizer. Defaults to 2e-5.
            warmup_proportion (float, optional): Proportion of training to
                perform linear learning rate warmup for. E.g., 0.1 = 10% of
                training. Defaults to None. 
            cache_dir (str, optional): Location of BERT's cache directory.
                Defaults to "./temp".

        Returns:
            None
        
        Raises:
            PermissionError:  If the running process doesn't have permission to create the cache dir
            Exception: If other unexpected errors occur during directory creation.
            
        """

        if not os.path.isdir(cache_dir):
            try:
                os.mkdir(cache_dir)
            except PermissionError as error:
                print(
                    "Please check permission if you can create or access the cache dir {0}. Additonal error message: {1}".format(
                        cache_dir, error
                    )
                )
                raise
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise

        self.cache_dir = cache_dir
        self.language = language
        self.to_lower = to_lower

        self.max_len = max_seq_length
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.num_epochs = num_epochs

        self.train_size = train_size
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.saved_model = None
        self.id_to_category = None
        self.saved_luis_model = None

        self.tokenizer = Tokenizer(
            self.language, self.to_lower, cache_dir=self.cache_dir
        )

    def get_train_dataframe(self, luis_model_file):
        """ Prepare training dataframe from luis model file

        Args:
            luis_model_file (str): file path of the luis model file for training 

        Returns:
            dict: json object of the luis model
            pandas.DataFrame: the training utterances and their labels
            dict:  the file path to save the dictionary  which maps
                the ids to categories from label encoder

        """

        luis_model = None
        with codecs.open(luis_model_file, "r", encoding="utf8") as fd:
            luis_model = json.load(fd)
        utterances = []
        utterances_text = [i.text for i in utterances]
        labels = [i["intent"] for i in luis_model["utterances"]]
        utterances_text = [i["text"] for i in luis_model["utterances"]]

        train_df = pd.DataFrame({"text": utterances_text})
        train_df["label"] = labels
        train_df["category_id"] = train_df["label"].factorize()[0]
        category_id_df = (
            train_df[["label", "category_id"]]
            .drop_duplicates()
            .sort_values("category_id")
        )
        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[["category_id", "label"]].values)
        return luis_model, train_df, id_to_category

    def load(self, model_file, id_to_category_file):
        """ Loads a saved SequenceClassifier model and a dictionary which maps 
        id to category from label encoder.

        Args:
            model_file (str): the file path of the SequenceClassifier model
            id_to_category_file (str): the file path of the the dictionary which maps
                ids to categories from label encoder

        """

        if torch.cuda.is_available():
            self.saved_model = torch.load(model_file)
            self.id_to_category = torch.load(id_to_category_file)
        else:
            self.saved_model = torch.load(model_file, map_location="cpu")
            self.id_to_category = torch.load(
                id_to_category_file, map_location="cpu"
            )

    def save(self, classifier_file, model_file=None, id_to_category_file=None):
        """ Saves the trained intent classifier model, and also saves the its corresponding
             SequenceClassifier model and the dictionary which maps ids to categories from label encoder.

        Args:
            classifier_file (str): the file path to save the intent classifier instance
            model_file (str, optional): the file path to save the SequenceClassifier model
            id_to_category_file (str, optional): the file path to save the dictionary 
                which maps ids to categories from label encoder

        """

        torch.save(self, classifier_file)
        if model_file:
            torch.save(self.saved_model, model_file)
        if id_to_category_file:
            torch.save(self.id_to_category, id_to_category_file)

    def train(self, luis_model_file):
        """ Fine-tunes the BERT classifier using the given luis model file.

        Args:
            luis_model_file (str): file path of the luis model file for training.

        """

        self.saved_luis_model, self.train_df, self.id_to_category = self.get_train_dataframe(
            luis_model_file
        )
        if self.train_size < 1.0:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=1 - self.train_size, random_state=0
            )
            for train_index, test_index in sss.split(
                self.train_df["text"], self.train_df["category_id"]
            ):
                df_train = self.train_df.iloc[train_index, :]
                df_test = self.train_df.iloc[test_index, :]
        else:  # don't use split as the dataset can be really small
            df_train = self.train_df
            df_test = self.train_df
        tokens_train = self.tokenizer.tokenize(list(df_train["text"]))
        tokens_test = self.tokenizer.tokenize(list(df_test["text"]))
        tokens_train, mask_train, _ = self.tokenizer.preprocess_classification_tokens(
            tokens_train, self.max_len
        )
        tokens_test, mask_test, _ = self.tokenizer.preprocess_classification_tokens(
            tokens_test, self.max_len
        )
        classifier = BERTSequenceClassifier(
            language=self.language,
            num_labels=len(self.id_to_category.keys()),
            cache_dir=self.cache_dir,
        )
        with Timer() as t:
            classifier.fit(
                token_ids=tokens_train,
                input_mask=mask_train,
                labels=df_train["category_id"].values,
                num_gpus=self.num_gpus,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                warmup_proportion=self.warmup_proportion,
                verbose=True,
            )
        logger.info("[Training time: {:.3f} hrs]".format(t.interval / 3600))
        preds = classifier.predict(
            token_ids=tokens_test,
            input_mask=mask_test,
            num_gpus=self.num_gpus,
            batch_size=self.batch_size,
        )
        target_names = [
            self.id_to_category[key]
            for key in sorted(self.id_to_category.keys())
        ]
        logger.info(
            classification_report(
                df_test["category_id"].values, preds, target_names=target_names
            )
        )
        self.saved_model = classifier

    def copy_for_predict(self, external_model):
        """copy an external intent classifier so predict function 
        can be updated from the source code.

        Args: 
            external_model (obj): an trained instance of BERTIntentClassifier

"""

        self.tokenizer = external_model.tokenizer
        self.num_gpus = external_model.num_gpus
        self.batch_size = external_model.batch_size
        self.id_to_category = external_model.id_to_category
        self.saved_model = external_model.saved_model
        self.saved_luis_model = external_model.saved_luis_model

    def predict(self, text):
        """ predict the intent of the text based on the trained model.

        Args:
            text (str):  the input text

        Returns:
            dict: a dictionary of the top scoring intent and also the intent ranking of 
                all intents in the trained model with scores.

        """

        tokens_test = self.tokenizer.tokenize(list([text]))
        tokens_test, mask_test, _ = self.tokenizer.preprocess_classification_tokens(
            tokens_test, self.max_len
        )
        preds = self.saved_model.predict(
            token_ids=tokens_test,
            input_mask=mask_test,
            num_gpus=self.num_gpus,
            batch_size=self.batch_size,
            probabilities=True,
        )
        max_index = preds.classes[0]
        probabilities = preds.probabilities[0]
        sorted_index = sorted(
            range(len(probabilities)),
            key=lambda k: probabilities[k],
            reverse=True,
        )
        intents = []
        for i in sorted_index:
            intents.append(
                {
                    "name": self.id_to_category[i],
                    "confidence": probabilities[i],
                }
            )
        top_intent = {
            "name": self.id_to_category[max_index],
            "confidence": probabilities[max_index],
        }
        result = {"intent": top_intent, "intent_ranking": intents}
        return result
