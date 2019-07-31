# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import torch

from utils_nlp.apps.luis.bert_entity_model import BERTEntityExtractor
from utils_nlp.models.bert.common import Language
from utils_nlp.dataset.url_utils import maybe_download

URL = "https://raw.githubusercontent.com/microsoft/LUIS-Samples/master/examples/example-app-models/custom/HomeAutomation.json"
FILE_NAME = URL.split("/")[-1]

EXTRACTOR_FILE_NAME = '-'.join(['bert-entity_extractor', FILE_NAME.split('.')[0]])

@pytest.fixture()
def luis_model_file(tmpdir):
    print(FILE_NAME)
    maybe_download(URL, FILE_NAME, tmpdir)
    assert os.path.exists(os.path.join(tmpdir, FILE_NAME))
    return os.path.join(tmpdir, FILE_NAME)

def test_bert_entity_model_training(luis_model_file):
    print(luis_model_file)
    language = Language.ENGLISHCASED
    # choose num_train_epoch 2 to train fast
    extractor = BERTEntityExtractor(language=language, num_epochs=2, batch_size = 8)
    extractor.train(luis_model_file)
    extractor.save(EXTRACTOR_FILE_NAME)

    assert extractor.saved_model is not None

def test_bert_intent_model_predict(tmpdir):
    language = Language.ENGLISHCASED
    if torch.cuda.is_available():
        extractor = torch.load(EXTRACTOR_FILE_NAME)
    else:
        extractor = torch.load(EXTRACTOR_FILE_NAME, map_location='cpu')
    result = extractor.predict("turn living room lights to green")
    assert len(result) > 0
    assert result[0].entity ==  "ContentName"
    assert result[0].start_pos ==  27
    assert result[0].entity ==  27 + len("green") -1

