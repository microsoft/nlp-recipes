import os
import random
import pandas as pd
from utils_nlp.dataset.ner_utils import preprocess_conll


FILES = {
    "train": "MSRA/msra-bakeoff3-training-utf8.2col",
    "test": "MSRA/bakeoff3_goldstandard.txt",
}
ENCODINGS = {"train": "utf8", "test": "gbk"}


def load_pandas_df(local_cache_path="./", file_split="test"):
    file_path = os.path.join(local_cache_path, FILES[file_split])
    encoding = ENCODINGS[file_split]

    with open(file_path, encoding=encoding) as file_path:
        text = file_path.read()

    # Add line break after punctuations indicating end of sentence in Chinese
    text = text.replace("。 0", "。 0\n")
    text = text.replace("？ 0", "？ 0\n")
    text = text.replace("！ 0", "！ 0\n")

    sentence_list, labels_list = preprocess_conll(text, file_split)

    sentence_and_labels = list(zip(sentence_list, labels_list))
    random.seed(100)
    random.shuffle(sentence_and_labels)
    sentence_list[:], labels_list[:] = zip(*sentence_and_labels)

    labels_list = [
        ["O" if label == "0" else label for label in labels]
        for labels in labels_list
    ]

    df = pd.DataFrame({"sentence": sentence_list, "labels": labels_list})

    return df


def get_unique_labels():
    return ["O", "B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER"]
