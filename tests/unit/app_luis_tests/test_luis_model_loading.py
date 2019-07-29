# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import pytest

from utils_nlp.apps.luis.utterance import Utterance

def test_loading_luis_file():
    with open("./FoodTruck.json") as fd:
        luis_model = json.load(fd)
        utterances = []
        for utterance_obj in luis_model["utterances"]:
            utterances.append(Utterance(utterance_obj))
        assert len(utterances) > 0
