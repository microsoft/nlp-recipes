# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path

#from models.gensen.localcode import train
from utils_nlp.model.gensen_utils import gensen_preprocess


class GenSenClassifier:
    """ GenSen Classifier that trains a model on server NLP tasks.

    learning_rate (str): The learning rate for the model.

    config_filepath (str) : Configuration file that is used to train the model. This
    specifies the batch size, directories to load and save the model.

    cache_dir (str) : Location of GenSen's data directory.

    """

    def __init__(self, config_filepath, learning_rate=0.0001, cache_dir="."):
        self.learning_rate = learning_rate
        self.config_filepath = config_filepath
        self.cache_dir = os.path.join(cache_dir, "clean/snli_1.0/")

    def _validate_params(self):
        """Validate input params."""

        if not isinstance(self.learning_rate, float) or (
                self.learning_rate <= 0.0
        ):
            raise ValueError(
                "Learning rate must be of type float and greater than 0"
            )

        if not os.path.exists(self.config_filepath):
            raise FileNotFoundError("Provided config file does not exist!")

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

    def fit(self, train_df, dev_df):

        """ Method to train the Gensen model.

        Args:
            train_df: A dataframe containing tokenized sentences from the training set.
            dev_df: A dataframe containing tokenized sentences from the validation set.

        """

        self._validate_params()
        self._get_gensen_tokens(train_df, dev_df)
        print(self.cache_dir)
        # train.train(
        #    data_folder=self.cache_dir,
        #    config_file_path=self.config_filepath,
        #    learning_rate=self.learning_rate,
        # )

    def eval(self, test_df):

        """

        Method to evaluate the model on the test dataset. This uses SentEval utils.
        Returns: None

        """

        pass
