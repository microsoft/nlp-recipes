# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from utils_nlp.models.bert.common import Language
from utils_nlp.models.bert.sequence_classification import (
    BERTSequenceClassifier,
)


@pytest.fixture()
def data():
    return (
        ["hi", "hello", "what's wrong with us", "can I leave?"],
        [0, 0, 1, 2],
    )

@pytest.mark.cpu
def test_classifier(bert_english_tokenizer, data):
    tokens = bert_english_tokenizer.tokenize(data[0])
    tokens, mask, _ = bert_english_tokenizer.preprocess_classification_tokens(
        tokens, max_len=10
    )

    classifier = BERTSequenceClassifier(
        language=Language.ENGLISHCASED, num_labels=3
    )
    classifier.fit(
        token_ids=tokens,
        input_mask=mask,
        labels=data[1],
        num_gpus=0,
        num_epochs=1,
        batch_size=2,
        verbose=True,
    )

    preds = classifier.predict(
        token_ids=tokens, input_mask=mask, num_gpus=0, batch_size=2
    )
    assert len(preds) == len(data[1])

@pytest.mark.gpu
def test_classifier_gpu_train_cpu_predict(bert_english_tokenizer, data):
    tokens = bert_english_tokenizer.tokenize(data[0])
    tokens, mask, _ = bert_english_tokenizer.preprocess_classification_tokens(
        tokens, max_len=10
    )

    classifier = BERTSequenceClassifier(
        language=Language.ENGLISHCASED, num_labels=3
    )
    classifier.fit(
        token_ids=tokens,
        input_mask=mask,
        labels=data[1],
        num_gpus=1,
        num_epochs=1,
        batch_size=2,
        verbose=True,
    )

    assert next(classifier.model.parameters()).is_cuda is True 
    # gpu prediction, no model move
    preds = classifier.predict(
        token_ids=tokens, input_mask=mask, num_gpus=1, batch_size=2
    )
    assert len(preds) == len(data[1])
    # cpu prediction, need model move
    assert next(classifier.model.parameters()).is_cuda is True
    preds = classifier.predict(
        token_ids=tokens, input_mask=mask, num_gpus=0, batch_size=2
    )
    assert next(classifier.model.parameters()).is_cuda is False
