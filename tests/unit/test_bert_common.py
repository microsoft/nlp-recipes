import os
import sys

nlp_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.bert.common import Tokenizer

INPUT_TEXT = ["Sarah is studying in the library."]
INPUT_LABELS = [["I-PER", "O", "O", "O", "O", "I-LOC"]]

UNIQUE_LABELS = ["O", "I-LOC", "I-MISC", "I-PER", "I-ORG", "X"]
LABEL_MAP = {label: i for i, label in enumerate(UNIQUE_LABELS)}


def test_tokenizer_preprocess_ner_tokens():
    pass
    # tokenizer = Tokenizer()

    # preprocessed_tokens = tokenizer.preprocess_ner_tokens(
    #     text=INPUT_TEXT, labels=INPUT_LABELS, label_map=LABEL_MAP
    # )


def test_create_data_loader():
    pass
