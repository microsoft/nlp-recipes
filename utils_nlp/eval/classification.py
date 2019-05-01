from sklearn.metrics import accuracy_score, precision_score, recall_score


def eval_classification(actual, predicted, round_decimals=4):
    return {
        "accuracy": accuracy_score(actual, predicted).round(round_decimals),
        "precision": list(
            precision_score(actual, predicted, average=None).round(
                round_decimals
            )
        ),
        "recall": list(
            recall_score(actual, predicted, average=None).round(round_decimals)
        ),
    }
