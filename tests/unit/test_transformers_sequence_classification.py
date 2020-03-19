# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import pandas as pd

from utils_nlp.models.transformers.sequence_classification import (
    SequenceClassifier,
    Processor,
)
from utils_nlp.common.pytorch_utils import dataloader_from_dataset


@pytest.fixture()
def data():
    return (["hi", "hello", "what's wrong with us", "can I leave?"], [0, 0, 1, 2])


@pytest.mark.cpu
def test_classifier(data, tmpdir):

    df = pd.DataFrame({"text": data[0], "label": data[1]})
    num_labels = len(pd.unique(data[1]))
    model_name = "bert-base-uncased"
    processor = Processor(model_name=model_name, cache_dir=tmpdir)
    ds = processor.dataset_from_dataframe(df, "text", "label")
    dl = dataloader_from_dataset(ds, batch_size=2, num_gpus=0, shuffle=True)
    classifier = SequenceClassifier(
        model_name=model_name, num_labels=num_labels, cache_dir=tmpdir
    )
    classifier.fit(train_dataloader=dl, num_epochs=1, num_gpus=0, verbose=False)
    preds = classifier.predict(dl, num_gpus=0, verbose=False)
    assert len(preds) == len(data[1])


@pytest.mark.gpu
def test_classifier_gpu_train_cpu_predict(data, tmpdir):

    df = pd.DataFrame({"text": data[0], "label": data[1]})
    num_labels = len(pd.unique(data[1]))
    model_name = "bert-base-uncased"
    processor = Processor(model_name=model_name, cache_dir=tmpdir)
    ds = processor.dataset_from_dataframe(df, "text", "label")
    dl = dataloader_from_dataset(ds, batch_size=2, num_gpus=1, shuffle=True)
    classifier = SequenceClassifier(
        model_name=model_name, num_labels=num_labels, cache_dir=tmpdir
    )
    classifier.fit(train_dataloader=dl, num_epochs=1, num_gpus=1, verbose=False)

    # gpu prediction, no model move
    preds = classifier.predict(dl, num_gpus=1, verbose=False)
    assert len(preds) == len(data[1])
    # cpu prediction, need model move
    assert next(classifier.model.parameters()).is_cuda is True
    preds = classifier.predict(dl, num_gpus=0, verbose=False)
    assert next(classifier.model.parameters()).is_cuda is False
