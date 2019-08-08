# NLP Utilities

This module (utils_nlp) contains functions to simplify common tasks used when developing and evaluating NLP systems. A short description of the sub-modules is provided below. For more details about what functions are available and how to use them, please review the doc-strings provided with the code.

## Sub-Modules

### [AzureML](azureml)

The AzureML submodule contains utilities to connect to a workspace, train, tune and operationalize NLP systems at scale using AzureML.

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

This submodule contains high-level utilities for defining constants used in most algorithms as well as helper functions for managing aspects of different frameworks like pytorch.

### [Dataset](dataset)
Dataset includes helper functions for interacting with different datasets and formatting them appropriately as well as utilities for splitting data for training / testing.

#### Data Loading
There are dataloaders for several datasets. For example, the snli module will allow you to load a dataframe in pandas from the SNLI dataset, with the option to set the number of rows to load in order to test algorithms and evaluate performance benchmarks. Information on the datasets used in the repo can be found [here](https://github.com/microsoft/nlp/tree/staging/utils_nlp/dataset#datasets).

Most datasets may be split into `train`, `dev`, and `test`.

```python
from utils_nlp.dataset.snli import load_pandas_df

df = load_pandas_df(DATA_FOLDER, file_split ="train", nrows = 1000)
```

### [Evaluation (Eval)](eval)
The evaluation (eval) submodule includes functionality for computing eturns common classification evaluation metrics like accuracy, precision, recall, and f1 scores for classification scenarios, normalizing and finding f1_scores for different datasets like SQuAD, as well as logging the means and other coefficients for datasets like senteval.

### [Models](models)
The models submodule contains implementations of various algorithms that can be used in addition to external packages to evaluate and develop new natural language processing systems. A description of which algorithms are used in each scenario can be found on [this table](../README.md#content)

This includes:
* BERT
* GenSen
* Pretrained embeddings (Word2Vec,
fastText,
GloVe)
* Pytorch's conditional Gated Recurrent Unit (GRU)

### [Interpreter](interpreter)
The interpreter submodule contains implementations to explain hidden states of models. It is a code implementation of the paper [Towards a Deep and Unified Understanding of Deep Neural Models in NLP](http://proceedings.mlr.press/v97/guan19a/guan19a.pdf).