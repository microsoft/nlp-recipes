# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
import os

from sklearn.metrics.pairwise import cosine_similarity

from utils_nlp.model.gensen import train_mlflow
from utils_nlp.model.gensen.create_gensen_model import (
    create_multiseq2seq_model,
)
from utils_nlp.model.gensen.gensen import GenSenSingle
from utils_nlp.model.gensen.gensen_utils import gensen_preprocess


class GenSenClassifier:
    """ GenSen Classifier that trains a model on server NLP tasks.

    learning_rate (str): The learning rate for the model.

    config_file (str) : Configuration file that is used to train the model. This
    specifies the batch size, directories to load and save the model.

    cache_dir (str) : Location of GenSen's data directory.

    """

    def __init__(
            self,
            config_file,
            pretrained_embedding_path,
            learning_rate=0.0001,
            cache_dir=".",
    ):
        self.learning_rate = learning_rate
        self.config_file = config_file
        self.cache_dir = cache_dir
        self.pretrained_embedding_path = pretrained_embedding_path
        self.model_name = "gensen_multiseq2seq"

    def _validate_params(self):
        """Validate input params."""

        if not isinstance(self.learning_rate, float) or (
                self.learning_rate <= 0.0
        ):
            raise ValueError(
                "Learning rate must be of type float and greater than 0"
            )

        assert os.path.isfile(self.pretrained_embedding_path)

        try:
            f = open(self.config_file)
            self.config = self._read_config(self.config_file)
            f.close()
        except FileNotFoundError:
            print("Provided config file does not exist!")

    def _get_gensen_tokens(self, train_df=None, dev_df=None, test_df=None):
        """

        Args:
            train_df(pd.Dataframe): A dataframe containing tokenized sentences from
        the training set.
            dev_df(pd.Dataframe): A dataframe containing tokenized
        sentences from the validation set.
            test_df(pd.Dataframe): A dataframe containing tokenized sentences from the
        test set.

        Returns:

        """
        return gensen_preprocess(train_df, dev_df, test_df, self.cache_dir)

    @staticmethod
    def _read_config(config_file):
        """ Read JSON config.

        Args:
            config_file: Path to the config file.

        Returns(dict): The loaded json file as python object

        """
        json_object = json.load(open(config_file, "r", encoding="utf-8"))
        return json_object

    def _create_multiseq2seq_model(self):
        """ Method that creates a GenSen model from a MultiSeq2Seq model."""

        create_multiseq2seq_model(
            save_folder=os.path.join(
                self.cache_dir, self.config["data"]["save_dir"]
            ),
            save_name=self.model_name,
            trained_model_folder=os.path.join(
                self.cache_dir, self.config["data"]["save_dir"]
            ),
        )

    def fit(self, train_df, dev_df, test_df):

        """ Method to train the Gensen model.

        Args:
            train_df: A dataframe containing tokenized sentences from the training set.
            dev_df: A dataframe containing tokenized sentences from the validation set.
            test_df: A dataframe containing tokenized sentences from the test set.
        """

        self._validate_params()
        self.cache_dir = self._get_gensen_tokens(train_df, dev_df, test_df)

        train_mlflow.train(
            data_folder=os.path.abspath(self.cache_dir),
            config=self.config,
            learning_rate=self.learning_rate,
        )

        self._create_multiseq2seq_model()

    def predict(self, sentences):

        """

        Method to predict the model on the test dataset. This uses SentEval utils.

        Args:
            sentences(list) : List of sentences.

        Returns(array): A pairwise cosine similarity for the sentences provided based
        on their gensen vector representations.

        """

        self._validate_params()

        # Use only if you have the model trained and saved.
        # self.cache_dir = os.path.join(self.cache_dir, "clean/snli_1.0")
        self._create_multiseq2seq_model()

        gensen_model = GenSenSingle(
            model_folder=os.path.join(
                self.cache_dir, self.config["data"]["save_dir"]
            ),
            filename_prefix=self.model_name,
            pretrained_emb=self.pretrained_embedding_path,
        )

        reps_h, reps_h_t = gensen_model.get_representation(
            sentences, pool="last", return_numpy=True
        )

        return cosine_similarity(reps_h_t)
