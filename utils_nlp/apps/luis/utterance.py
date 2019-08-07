# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from hashlib import md5
import json
import string
from utils_nlp.apps.luis.entity import Entity


class Utterance:
    """
        class definition for LUIS utterance
    """

    text = ""
    intent = ""
    entities = []
    processed = None

    def __init__(self, obj):
        self.text = obj["text"]
        self.intent = obj["intent"]
        self.entities = []
        for entity_obj in obj["entities"]:
            new_obj = entity_obj
            # add value to enable easy comparison during test
            new_obj["value"] = self.text[new_obj["startPos"] : new_obj["endPos"] + 1]
            self.entities.append(Entity(new_obj))

    def __eq__(self, other):
        return (
            self.processed == other.processed
            and self.intent.lower() == other.intent.lower()
        )

    def as_dict(self):
        return {
            "text": self.text,
            "intent": self.intent,
            "entities": [x.__dict__ for x in self.entities],
        }

    def __hash__(self):
        return md5(
            json.dumps(
                {"text": self.processed, "intent": self.intent.lower()}
            ).encode("utf-8")
        )
