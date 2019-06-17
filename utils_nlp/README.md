# Utils to help keep the notebooks focused on NLP

## BERT
Provides utils for loading pretrained bert token and sequence classifiers.
As well, utils for tokenizing inputs for BERT based models.

## Common
Contains a Timer class to help time snippets of code
Indented code:
    with Timer() as t:
        # code snipper
    assert t.interval < 1

## PyTorch
Utils for configuring CPU vs GPU for PyTorch models and training.

## Mlflow
utils_nlp.mlflow.set_mlflow_tracking_uri, sets the tracking uri to the current
AzureML workspace if available for long term tracking. Otherwise, sets the tracking uri to the
current working directory to avoid issues that arise from directory changes for the local store.

Indented code:

    from utils_nlp.mlflow import set_mlflow_tracking_uri

    # sets the tracking uri to the AzureML workspace's if azureml-mlflow is pip installed.
    set_mlflow_tracking_uri()  

    # sets the tracking uri to the current directory if no workspace is available.
    set_mlflow_tracking_uri()  

    # sets the tracking uri to my_cool_tracking_uri
    set_mlflow_tracking_uri(my_cool_tracking_uri)  

    # sets the tracking uri to my_cool_workspace's tracking uri
    set_mlflow_tracking_uri(workspace=my_cool_workspace)  

## Dataset
Utils for downloading datasets. Current datasets include:
1. MultNLI
2. XNLI
3. DAC
4. Wiki Gold
5. Yahoo Answers
6. MSRP

Also includes STSBenchmark tools and dataset preprocessing utils.

## Eval
Utils for calculating metrics for actual vs predicted values. Model wrapper for Senteval


## Pretrained Embeddings
Loaders for fasttext, gloVe, and word2vec word embeddings.

## AzureML
Utils for loading an AzureML workspace from the environment.
