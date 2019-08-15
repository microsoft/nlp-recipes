# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import pandas as pd
def generate_confusion_matrix(y_true, y_pred,labels):
    """Function that prints out a graphical representation of confusion matrix using Seaborn Heatmap

    Args:
        y_true: True labels from dataset
        y_pred: Predicted labels from the models
        labels: A list of labels

    """
    conf_matrix=np.array(confusion_matrix(y_true, y_pred))
    conf_matrix=np.divide(conf_matrix.astype('float'), conf_matrix.sum())
    conf_dataframe = pd.DataFrame(conf_matrix,labels,labels)
    dimensions_plot = (8, 5)
    fig, ax = pyplot.subplots(figsize=dimensions_plot)
    sn.set(font_scale=1.1)
    ax=sn.heatmap(conf_dataframe,cmap="Blues", annot=True,annot_kws={"size": 16})
    ax.set(xlabel='True Labels', ylabel='Predicted Labels')