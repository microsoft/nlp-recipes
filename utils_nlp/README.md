# NLP Utilities

Modern NLP research and development can involve tedious tasks ranging from data loading, dataset understanding,  model development, model evaluation to productionize a trained NLP model. Recognizing the need of simplying these tedious tasks, we developed this module (**utils_nlp**) to provide a wide spectrum of classes, functions and utilities. Adoption of this module can greately speed up the development work and sample notebooks in [Examples](../examples) folder can demonstrate this.  The following provides a short description of the sub-modules. For more details about what functions/classes/utitilies are available and how to use them, please review the doc-strings provided with the code and see the sample notebooks in [Examples](../examples) folder.

## Submodules

### [AzureML](azureml)

The AzureML submodule contains utilities to connect to an Azure Machine Learning workspace, train, tune and operationalize NLP systems at scale using AzureML.

```python
from utils_nlp.azureml.azureml_utils import get_or_create_workspace

###Note: you do not need to fill in these values if you have a config.json in the same folder as this notebook
ws = get_or_create_workspace(
    config_path=config_path,
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
    workspace_region=workspace_region,
)
```

### [Common](common)

This submodule contains high-level utilities that are commonly used in multiple algorithms as well as helper functions for managing frameworks like pytorch.

### [Dataset](dataset)
This submodule includes helper functions for interacting with well-known datasets,  utility functions to process datasets for different NLP tasks, as well as utilities for splitting data for training/testing. For example, the [snli module](snli.py) will allow you to load a dataframe in pandas from the  Stanford Natural Language Inference (SNLI) Corpus dataset, with the option to set the number of rows to load in order to test algorithms and evaluate performance benchmarks. Information on the datasets used in the repo can be found [here](https://github.com/microsoft/nlp-recipes/tree/staging/utils_nlp/dataset#datasets).

Most datasets may be split into `train`, `dev`, and `test`.

```python
from utils_nlp.dataset.snli import load_pandas_df

df = load_pandas_df(DATA_FOLDER, file_split ="train", nrows = 1000)
```

### [Evaluation](eval)
The *eval* submodule includes functionalities for computing common classification evaluation metrics like accuracy, precision, recall, and f1 scores for classification scenarios. It also includes metric utitlities for normalizing and finding f1_scores for [The Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/), and utilities to log the means and other coefficients in evaluating the quality of sentence embedding.

### [Models](models)
The models submodule contains implementations of various algorithms that can be used in addition to external packages to evaluate and develop new natural language processing systems. A description of which algorithms are used in each scenario can be found on [this table](../README.md#content).

A few highlights are
* BERT
* GenSen
* XLNet


### [Model Explainability](interpreter)
The interpreter submodule contains utils that help explain or diagnose models, such as interpreting layers of a neural network.
