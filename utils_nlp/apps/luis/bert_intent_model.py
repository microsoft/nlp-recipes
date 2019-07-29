
import re
import codecs
import json
import logging
import os
import sys

import pandas as pd
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report

from utils_nlp.models.bert.sequence_classification import BERTSequenceClassifier
from utils_nlp.models.bert.common import Language, Tokenizer
from utils_nlp.common.timer import Timer


logger = logging.getLogger(__name__)


class BERTIntentClassifier:
    def __init__(
        self,
        language,
        to_lower=False,
        num_epochs=10,
        max_seq_length=100,
        batch_size=16,
        train_size=0.8,
        learning_rate=3e-5,
        cache_dir="./temp",
        num_gpus=None,
    ):
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        if not os.path.isdir(cache_dir):
            raise Exception(
                "Please check permission if you can create or access the cache dir {0}".format(
                    cache_dir
                )
            )
        self.cache_dir = cache_dir
        self.language = language
        self.to_lower = to_lower

        self.max_len = max_seq_length
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.num_epochs = num_epochs

        self.train_size = train_size

        self.saved_model = None
        self.id_to_category = None
        self.saved_luis_model = None

        self.tokenizer = Tokenizer(
            self.language, self.to_lower, cache_dir=self.cache_dir
        )

    def get_train_dataframe(self, luis_model_file):
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
        if torch.cuda.is_available():
            self.saved_model = torch.load(model_file)
            self.id_to_category = torch.load(id_to_category_file)
        else:
            self.saved_model = torch.load(model_file, map_location="cpu")
            self.id_to_category = torch.load(
                id_to_category_file, map_location="cpu"
            )

    def save(self, classifier_file, model_file=None, id_to_category_file=None):
        torch.save(self, classifier_file)
        if model_file:
            torch.save(self.saved_model, model_file)
        if id_to_category_file:
            torch.save(self.id_to_category, id_to_category_file)

    def train(self, luis_model_file):
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
        else: #don't use split as the dataset can be really small
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
        self.tokenizer = external_model.tokenizer
        self.num_gpus = external_model.num_gpus
        self.batch_size = external_model.batch_size
        self.id_to_category = external_model.id_to_category
        self.saved_model = external_model.saved_model
        self.saved_luis_model = external_model.saved_luis_model

    def predict(self, text, progress=True):
        tokens_test = self.tokenizer.tokenize(
            list([text]),
        )
        tokens_test, mask_test, _ = self.tokenizer.preprocess_classification_tokens(
            tokens_test, self.max_len
        )
        preds = self.saved_model.predict(
            token_ids=tokens_test,
            input_mask=mask_test,
            num_gpus=self.num_gpus,
            batch_size=self.batch_size,
            probabilities=True
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
