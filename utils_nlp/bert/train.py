
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import logging
import argparse
import pickle

from sklearn.metrics import classification_report

from utils_nlp.bert.common import Language
from utils_nlp.bert.sequence_classification_distributed import BERTSequenceDistClassifier
from utils_nlp.common.timer import Timer

BATCH_SIZE = 32
NUM_GPUS = 2
NUM_EPOCHS = 1
LABELS = ["telephone", "government", "travel", "slate", "fiction"]

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--input_train_dir", type=str, help="Training data")
parser.add_argument("--input_test_dir", type=str, help="Test data")
parser.add_argument("--result_dir", type=str, help="Results directory containing confidence report")
parser.add_argument("--result_file", type=str, help="File name for confidence report")

args = parser.parse_args()
train_dir = args.input_train_dir
test_dir = args.input_test_dir
result_dir = args.result_dir
result_file = args.result_file

if result_dir is not None:
    os.makedirs(result_dir, exist_ok=True)
    logger.info("%s created" % result_dir)

# Train
classifier = BERTSequenceDistClassifier(
    language=Language.ENGLISH, num_labels=len(LABELS)
)
with Timer() as t:
    classifier.fit(
        train_dir,
        num_gpus=NUM_GPUS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True,
    )
logger.info("Training Time {}".format(t.interval / 3600))

# Predict
preds, labels_test = classifier.predict(
    test_dir, num_gpus=NUM_GPUS, batch_size=BATCH_SIZE
)
data = classification_report(labels_test, preds, target_names=LABELS)
with open(os.path.join(result_dir, result_file), 'wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
