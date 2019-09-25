# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utilities functions for computing general model evaluation metrics."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from numpy import corrcoef

from matplotlib import pyplot
import seaborn as sn
import numpy as np
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


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    normalize=False,
    title="Confusion matrix",
    plot_size=(8, 5),
    font_scale=1.1,
):
    """Function that prints out a graphical representation of confusion matrix using Seaborn Heatmap

    Args:
        y_true (1d array-like): True labels from dataset
        y_pred (1d array-like): Predicted labels from the models
        labels: A list of labels
        normalize (Bool, optional): Boolean to Set Row Normalization for Confusion Matrix
        title (String, optional): String that is the title of the plot
        plot_size (tuple, optional): Tuple of Plot Dimensions Default "(8, 5)"
        font_scale (float, optional): float type scale factor for font within plot
    """
    conf_matrix = np.array(confusion_matrix(y_true, y_pred))
    if normalize:
        conf_matrix = np.round(
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis], 3
        )
    conf_dataframe = pd.DataFrame(conf_matrix, labels, labels)
    fig, ax = pyplot.subplots(figsize=plot_size)
    sn.set(font_scale=font_scale)
    ax.set_title(title)
    ax = sn.heatmap(conf_dataframe, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt="g")
    ax.set(xlabel="Predicted Labels", ylabel="True Labels")
