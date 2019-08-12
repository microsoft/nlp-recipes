# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import json
from functools import total_ordering


@total_ordering
class Entity:
    """
       class definition for LUIS entity.
    """

    entity = None
    start_pos = -1
    end_pos = -1

    def __init__(self, obj):
        # self.words = obj['Words']
        self.start_pos = obj["startPos"]
        self.end_pos = obj["endPos"]
        self.entity = obj["entity"]
        self.value = None
        self.confidence = None
        self.resolution = None
        if "value" in obj:
            self.value = obj["value"]
        if "confidence" in obj:
            self.confidence = obj["confidence"]
        if "resolution" in obj:
            self.resolution = obj["resolution"]

    def __eq__(self, other):
        return (
            self.entity.lower() == other.entity.lower()
            and self.start_pos == other.start_pos
            and self.end_pos == other.end_pos
        )

    def __hash__(self):
        return hash(
            json.dumps(
                {
                    "startPos": self.start_pos,
                    "endPos": self.end_pos,
                    "entity": self.entity,
                }
            ).encode("utf-8")
        )

    def as_dict(self):
        return self.__dict__

    def __repr__(self):
        return repr(
            (
                self.entity,
                self.start_pos,
                self.end_pos,
                self.value,
                self.confidence,
                self.resolution,
            )
        )

    def __le__(self, other):

        if self.start_pos > other.start_pos:
            return False
        elif self.end_pos > other.end_pos:
            return False
        return True
