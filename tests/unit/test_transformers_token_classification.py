# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from utils_nlp.common.pytorch_utils import dataloader_from_dataset
from utils_nlp.models.transformers.named_entity_recognition import TokenClassificationProcessor, TokenClassifier


@pytest.mark.cpu
def test_token_classifier_fit_predict(tmpdir, ner_test_data):
    token_classifier = TokenClassifier(model_name="bert-base-uncased", num_labels=6, cache_dir=tmpdir)
    processor = TokenClassificationProcessor(model_name="bert-base-uncased", cache_dir=tmpdir)

    # test fit, no warmup
    train_dataset = processor.preprocess_for_bert(
        text=ner_test_data["INPUT_TEXT"], labels=ner_test_data["INPUT_LABELS"], label_map=ner_test_data["LABEL_MAP"],
    )
    train_dataloader = dataloader_from_dataset(train_dataset)
    token_classifier.fit(train_dataloader)

    # test predict, no labels
    _ = token_classifier.predict(train_dataloader, verbose=False)
