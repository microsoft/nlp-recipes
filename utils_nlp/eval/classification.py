# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def eval_classification(actual, predicted, round_decimals=4):
    """Returns common classification evaluation metrics.
    Args:
        actual (1d array-like): Array of actual values.
        predicted (1d array-like): Array of predicted values.
        round_decimals (int, optional): Number of decimal places. Defaults to 4.
    Returns:
        dict: A dictionary of evaluation metrics.
    """
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
        "f1": list(
            f1_score(actual, predicted, average=None).round(round_decimals)
        ),
    }
