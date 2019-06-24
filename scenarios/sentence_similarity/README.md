# Sentence Similarity

This folder contains examples and best practices, written in Jupyter notebooks, for building sentence similarity models. The scores can be used in a wide variety of applications, such as search/retrieval, nearest-neighbor or kernel-based classification methods, recommendation, and ranking tasks.

## What is sentence similarity

Sentence similarity or semantic textual similarity is to determine how similar two pieces of texts are and a measure of the degree to which two pieces of text express the same meaning. This can take the form of assigning a score from 1 to 5. Related tasks are paraphrase or duplicate identification. The common methods used for text similarity range from simple word-vector dot products to pairwise classification, and more recently, Siamese recurrent/convolutional neural networks with triplet loss functions.

Sentence similarity is normally calculated by the following two steps:

1. obtaining the embeddings of the sentences

2. taking the cosine similarity between them as shown in the following figure([Source](https://tfhub.dev/google/universal-sentence-encoder/1)):
    ![Sentence Similarity](https://nlpbp.blob.core.windows.net/images/example-similarity.png)

## Summary

The following summarizes each notebook for Sentence Similarity. Each notebook provides more details and guiding in principles on building state of the art models.

|Notebook|Runs Local|Description|
|---|---|---|
|[Creating a Baseline model](baseline_deep_dive.ipynb)| Yes| A baseline model is a basic solution that serves as a point of reference for comparing other models to. The baseline model's performance gives us an indication of how much better our models can perform relative to a naive approach.|
|Senteval |[local](senteval_local.ipynb), [AzureML](senteval_azureml.ipynb)|SentEval is a widely used benchmarking tool for evaluating general-purpose sentence embeddings. Running SentEval locally is easy, but not necessarily efficient depending on the model specs. We provide an example on how to do this efficiently in Azure Machine Learning Service. |
|[GenSen on AzureML](gensen_aml_deep_dive.ipynb_)| No | This notebook serves as an introduction to an end-to-end NLP solution for sentence similarity building one of the State of the Art models, GenSen, on the AzureML platform. We show the advantages of AzureML when training large NLP models with GPU.
