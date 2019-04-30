from sklearn.metrics import accuracy_score, precision_score, recall_score


def eval_classification(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, average=None),
        "recall": recall_score(actual, predicted, average=None),
    }

