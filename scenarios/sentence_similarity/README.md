# Sentence Similarity

This folder contains examples and best practices, written in Jupyter notebooks, for building sentence similarity models. The scores can be used in a wide variety of applications, such as search/retrieval, nearest-neighbor or kernel-based classification methods, recommendations, and ranking tasks.

## What is sentence similarity

Sentence similarity or semantic textual similarity is a measure of how similar two pieces of text are, or to what degree they express the same meaning. Related tasks include paraphrase or duplicate identification, search, and matching applications. The common methods used for text similarity range from simple word-vector dot products to pairwise classification, and more recently, deep neural networks.

Sentence similarity is normally calculated by the following two steps:

1. obtaining the embeddings of the sentences

2. taking the cosine similarity between them as shown in the following figure([source](https://tfhub.dev/google/universal-sentence-encoder/1)):

    ![Sentence Similarity](https://nlpbp.blob.core.windows.net/images/example-similarity.png)

## Summary

The following summarizes each notebook for Sentence Similarity. Each notebook provides more details and guiding in principles on building state of the art models.

|Notebook|Runs Local|Description|
|---|---|---|
|[Creating a Baseline model](baseline_deep_dive.ipynb)| Yes| A baseline model is a basic solution that serves as a point of reference for comparing other models to. The baseline model's performance gives us an indication of how much better our models can perform relative to a naive approach.|
|Senteval |[local](senteval_local.ipynb), [AzureML](senteval_azureml.ipynb)|SentEval is a widely used benchmarking tool for evaluating general-purpose sentence embeddings. Running SentEval locally is easy, but not necessarily efficient depending on the model specs. We provide an example on how to do this efficiently in Azure Machine Learning Service. |
|[GenSen on AzureML](gensen_aml_deep_dive.ipynb)| No | This notebook serves as an introduction to an end-to-end NLP solution for sentence similarity building one of the State of the Art models, GenSen, on the AzureML platform. We show the advantages of AzureML when training large NLP models with GPU.
|[Automated Machine Learning(AutoML) with Deployment on Azure Container Instance](automl_local_deployment_aci.ipynb)| Yes |This notebook shows users how to use AutoML on local machine and deploy the model as a webservice to Azure Container Instance(ACI) to get a sentence similarity score.
|[Google Universal Sentence Encoder with Azure Machine Learning Pipeline, AutoML with Deployment on Azure Kubernetes Service](automl_with_pipelines_deployment_aks.ipynb)| No | This notebook shows a user how to use AzureML pipelines and deploy the pipeline output model as a webservice to Azure Kubernetes Service which can be used as an end point to get sentence similarity scores.  
