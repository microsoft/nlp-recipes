# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from utils_nlp.models.xlnet.common import Language
from utils_nlp.models.xlnet.sequence_classification import (
    XLNetSequenceClassifier,
)


@pytest.fixture()
def data():
    return (
        ["hi", "hello", "what's wrong with us", "can I leave?"],
        [0, 0, 1, 2],
    )


def test_classifier(xlnet_english_tokenizer, data):
    token_ids, input_mask, segment_ids = xlnet_english_tokenizer.preprocess_classification_tokens(
        data[0], max_seq_length=10
    )

    classifier = XLNetSequenceClassifier(
        language=Language.ENGLISHCASED, num_labels=3
    )
    classifier.fit(
        token_ids=token_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        labels=data[1],    
        num_gpus=0,        
        num_epochs=1,
        batch_size=2,    
        verbose=True,
    )

    preds = classifier.predict(
        token_ids=token_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        num_gpus=0,
        batch_size=2,
        probabilities=False
    )
    assert len(preds) == len(data[1])