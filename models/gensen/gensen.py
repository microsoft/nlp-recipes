# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path

from models.gensen.localcode import train


class GenSenClassifier:
    """ GenSen Classifier that trains a model on server NLP tasks.

    learning_rate (str): The learning rate for the model.

    config_filepath (str) : Configuration file that is used to train the model. This
    specifies the batch size, directories to load and save the model.

    """

    def __init__(self, config_filepath, learning_rate=0.0001):
        self.learning_rate = learning_rate
        self.config_filepath = config_filepath

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

    def fit(self, data_path):

        """

        Args:
            data_path(str): Path to the folder containing the data.

        """

        self._validate_params()
        train.train(
            data_folder=data_path,
            config_file_path=self.config_filepath,
            learning_rate=self.learning_rate,
        )

    def eval(self):

        """

        Method to evaluate the model on the test dataset. This uses SentEval utils.
        Returns: None

        """

        pass
