# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import pytest

from utils_nlp.apps.luis.utterance import Utterance
from utils_nlp.dataset.url_utils import maybe_download

URL = "https://raw.githubusercontent.com/microsoft/LUIS-Samples/master/examples/example-app-models/custom/FoodTruck.json"
FILE_NAME = URL.split("/")[-1]


@pytest.fixture()
def luis_model(tmpdir):
    print(FILE_NAME)
    maybe_download(URL, FILE_NAME, tmpdir)
    assert os.path.exists(os.path.join(tmpdir, FILE_NAME))
    with open(os.path.join(tmpdir, FILE_NAME)) as fd:
        luis_model_obj = json.load(fd)
        return luis_model_obj
    return None


def test_loading_luis_file(luis_model):
    utterances = []
    for utterance_obj in luis_model["utterances"]:
        utterances.append(Utterance(utterance_obj))
    assert len(utterances) > 0
