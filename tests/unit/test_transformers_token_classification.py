# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from utils_nlp.common.pytorch_utils import dataloader_from_dataset
from utils_nlp.models.transformers.named_entity_recognition import (
    TokenClassificationProcessor,
    TokenClassifier,
)
from utils_nlp.models.transformers.common import MAX_SEQ_LEN


@pytest.mark.cpu
def test_token_classifier_fit_predict(tmpdir, ner_test_data):
    num_labels = 6
    max_seq_len = MAX_SEQ_LEN
    token_classifier = TokenClassifier(
        model_name="bert-base-uncased", num_labels=num_labels, cache_dir=tmpdir
    )
    processor = TokenClassificationProcessor(
        model_name="bert-base-uncased", cache_dir=tmpdir
    )

    # test fit, no warmup
    train_dataset = processor.preprocess(
        text=ner_test_data["INPUT_TEXT"],
        max_len=max_seq_len,
        labels=ner_test_data["INPUT_LABELS"],
        label_map=ner_test_data["LABEL_MAP"],
    )
    train_dataloader = dataloader_from_dataset(train_dataset)
    token_classifier.fit(train_dataloader)

    # test predict, no labels
    preds = token_classifier.predict(train_dataloader, verbose=False)
    assert preds.shape == (len(train_dataloader), MAX_SEQ_LEN, num_labels)
