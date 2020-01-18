# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from utils_nlp.common.pytorch_utils import dataloader_from_dataset
from utils_nlp.models.transformers.named_entity_recognition import TokenClassificationProcessor, TokenClassifier


def test_token_classifier_fit_predict(tmp_path, ner_test_data):
    token_classifier = TokenClassifier(num_labels=6, cache_dir=tmp_path)
    processor = TokenClassificationProcessor(cache_dir=tmp_path)

    # test fit, no warmup
    train_dataset = processor.preprocess_for_bert(
        text=ner_test_data["INPUT_TEXT"], labels=ner_test_data["INPUT_LABELS"], label_map=ner_test_data["LABEL_MAP"],
    )
    train_dataloader = dataloader_from_dataset(train_dataset)
    token_classifier.fit(train_dataloader)

    # test predict, no labels
    preds = token_classifier.predict(train_dataloader, verbose=False)
    assert len(preds) == len(ner_test_data["INPUT_LABELS"])
