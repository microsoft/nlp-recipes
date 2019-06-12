import os
import sys

nlp_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

# from utils_nlp.bert.token_classification import (
#     BERTTokenClassifier,
#     postprocess_token_labels,
# )
