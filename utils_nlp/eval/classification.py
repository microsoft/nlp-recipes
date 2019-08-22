# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utilities functions for computing general model evaluation metrics."""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from numpy import corrcoef
import pandas as pd


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
        "precision": list(precision_score(actual, predicted, average=None).round(round_decimals)),
        "recall": list(recall_score(actual, predicted, average=None).round(round_decimals)),
        "f1": list(f1_score(actual, predicted, average=None).round(round_decimals)),
    }


def compute_correlation_coefficients(x, y=None):
    """
    Compute Pearson product-moment correlation coefficients.

    Args:
        x: array_like
            A 1-D or 2-D array containing multiple variables and observations.
            Each row of `x` represents a variable, and each column a single
            observation of all those variables.

        y: array_like, optional
            An additional set of variables and observations. `y` has the same
            shape as `x`.

    Returns:
        pd.DataFrame : A pandas dataframe from the correlation coefficient matrix of the variables.
    """
    return pd.DataFrame(corrcoef(x, y))
