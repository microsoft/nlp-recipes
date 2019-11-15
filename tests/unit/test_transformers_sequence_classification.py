# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import pandas as pd

from utils_nlp.models.transformers.sequence_classification import SequenceClassifier, Processor


@pytest.fixture()
def data():
    return (["hi", "hello", "what's wrong with us", "can I leave?"], [0, 0, 1, 2])


@pytest.mark.cpu
def test_classifier(data, tmpdir):

    df = pd.DataFrame({"text": data[0], "label": data[1]})
    num_labels = len(pd.unique(data[1]))
    model_name = "bert-base-uncased"
    processor = Processor(model_name=model_name, cache_dir=tmpdir)
    train_dataloader = processor.create_dataloader_from_df(
        df, "text", "label", batch_size=2, num_gpus=0
    )
    classifier = SequenceClassifier(model_name=model_name, num_labels=num_labels, cache_dir=tmpdir)
    classifier.fit(train_dataloader=train_dataloader, num_epochs=1, num_gpus=0, verbose=False)
    preds = classifier.predict(train_dataloader, num_gpus=0, verbose=False)
    assert len(preds) == len(data[1])


@pytest.mark.gpu
def test_classifier_gpu_train_cpu_predict(data, tmpdir):

    df = pd.DataFrame({"text": data[0], "label": data[1]})
    num_labels = len(pd.unique(data[1]))
    model_name = "bert-base-uncased"
    processor = Processor(model_name=model_name, cache_dir=tmpdir)
    train_dataloader = processor.create_dataloader_from_df(
        df, "text", "label", batch_size=2, num_gpus=1
    )
    classifier = SequenceClassifier(model_name=model_name, num_labels=num_labels, cache_dir=tmpdir)
    classifier.fit(train_dataloader=train_dataloader, num_epochs=1, num_gpus=1, verbose=False)

    assert next(classifier.model.parameters()).is_cuda is True
    # gpu prediction, no model move
    preds = classifier.predict(train_dataloader, num_gpus=1, verbose=False)
    assert len(preds) == len(data[1])
    # cpu prediction, need model move
    assert next(classifier.model.parameters()).is_cuda is True
    preds = classifier.predict(train_dataloader, num_gpus=0, verbose=False)
    assert next(classifier.model.parameters()).is_cuda is False
