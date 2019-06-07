import sys
sys.path.append("../../")
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import utils_nlp.dataset.yahoo_answers as ya_dataset
from utils_nlp.eval.classification import eval_classification
from utils_nlp.bert.sequence_classification import SequenceClassifier
from utils_nlp.bert.common import Language, Tokenizer
from utils_nlp.common.timer import Timer
import torch
import torch.nn as nn
import numpy as np

DATA_FOLDER = "./temp"
BERT_CACHE_DIR = "./temp"
MAX_LEN = 10  # 250
BATCH_SIZE = 1
NUM_GPUS = 1
NUM_EPOCHS = 10

text_train = [
    ["this is a mouse", "hunt a cat"],
    ["my car", "BMW is nice"],
    ["my other car", "BMW is not nice"],
    ["my other car", "BMW is not nice"]
]

labels_train = [
    [[0, 4]],  # 4 non-clicks
    [[5, 0]],  # 5 clicks
    [[1, 9]],  # 1 click, 9 non-clicks
    [[1, 9]]  # 1 click, 9 non-clicks
]

tokenizer = Tokenizer(Language.ENGLISH, to_lower=True,
                      cache_dir=BERT_CACHE_DIR)

# tokenize
tokens_train = tokenizer.tokenize(text_train)

tokens_train, mask_train, token_type_ids = tokenizer.preprocess_classification_tokens(
    tokens_train, MAX_LEN
)

print(text_train)
print(tokens_train)
print(mask_train)
print(token_type_ids)
print(labels_train)

classifier = SequenceClassifier(
    language=Language.ENGLISH,
    num_labels=2,
    cache_dir=BERT_CACHE_DIR
)

# train
with Timer() as t:
    classifier.fit(
        token_ids=tokens_train,
        input_mask=mask_train,
        token_type_ids=token_type_ids,
        labels=labels_train,
        num_gpus=NUM_GPUS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True,
    )
print("[Training time: {:.3f} hrs]".format(t.interval / 3600))


for i in [1, 2, 4]:
    preds = classifier.predict(
        token_ids=tokens_train,
        input_mask=mask_train,
        token_type_ids=token_type_ids,
        num_gpus=NUM_GPUS,
        batch_size=i
    )

    print(preds)
