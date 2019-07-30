# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import torch

from utils_nlp.apps.luis.bert_intent_model import BERTIntentClassifier
from utils_nlp.models.bert.common import Language
from utils_nlp.dataset.url_utils import maybe_download

URL = "https://raw.githubusercontent.com/microsoft/LUIS-Samples/master/examples/example-app-models/custom/HomeAutomation.json"
FILE_NAME = URL.split("/")[-1]

CLASSIFIER_FILE_NAME = '-'.join(['bert-classifier', FILE_NAME.split('.')[0]])

@pytest.fixture()
def luis_model_file(tmpdir):
    print(FILE_NAME)
    maybe_download(URL, FILE_NAME, tmpdir)
    assert os.path.exists(os.path.join(tmpdir, FILE_NAME))
    return os.path.join(tmpdir, FILE_NAME)

def test_bert_intent_model_training(luis_model_file):
    print(luis_model_file)
    language = Language.ENGLISHCASED
    # choose num_train_epoch 2 to train fast
    classifier = BERTIntentClassifier(language=language, num_epochs=2, batch_size = 8, train_size=1.0)
    classifier.train(luis_model_file)
    classifier.save(CLASSIFIER_FILE_NAME)

    assert classifier.saved_model is not None
    assert classifier.id_to_category is not None

def test_bert_intent_model_predict(tmpdir):
    language = Language.ENGLISHCASED
    if torch.cuda.is_available():
        classifier = torch.load(CLASSIFIER_FILE_NAME)
    else:
        classifier = torch.load(CLASSIFIER_FILE_NAME, map_location='cpu')
    result = classifier.predict("what is the weather?")
    assert result['intent']["name"] == "GetCurrentWeather"
    assert result['intent']["confidence"] > 0.5

