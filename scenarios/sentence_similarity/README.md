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
|Senteval |[Local](senteval_local.ipynb), [AzureML](senteval_azureml.ipynb)|SentEval is a widely used benchmarking tool for evaluating general-purpose sentence embeddings. Running SentEval locally is easy, but not necessarily efficient depending on the model specs. We provide an example on how to do this efficiently in Azure Machine Learning Service. |
|Gensen | [Local](gensen_local.ipynb), [AzureML](gensen_aml_deep_dive.ipynb)|This notebook serves as an introduction to an end-to-end NLP solution for sentence similarity building one of the State of the Art models, GenSen. We provide two notebooks. One, which runs on the AzureML platform.  We show the advantages of AzureML when training large NLP models with GPU in this notebook. The other example walks through using a GPU enabled VM to train and score Gensen.|
|[Automated Machine Learning(AutoML) with Deployment on Azure Container Instance](automl_local_deployment_aci.ipynb)| Yes |This notebook shows users how to use AutoML on local machine and deploy the model as a webservice to Azure Container Instance(ACI) to get a sentence similarity score.
|[Google Universal Sentence Encoder with Azure Machine Learning Pipeline, AutoML with Deployment on Azure Kubernetes Service](automl_with_pipelines_deployment_aks.ipynb)| No | This notebook shows a user how to use AzureML pipelines and deploy the pipeline output model as a webservice to Azure Kubernetes Service which can be used as an end point to get sentence similarity scores.|

## Using GPU vs Azure ML Compute
We did a comparative study to make it easier for you to choose between a GPU enabled Azure VM 
and Azure ML compute. The table below provides the cost vs performance trade-off for 
each of the choices. 

* The total time in the table stands for the training time + setup time.
* Cost is the estimated cost of running the Azure ML Compute Job or the VM up-time.

**Please note:** These were the estimated cost for running these notebooks as of July 1. Please 
look at the [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) to see the most up to date pricing information. 

|---|Azure VM| AML 1 Node| AML 2 Nodes | AML 4 Nodes | AML 8 Nodes|
|---|---|---|---|---|---|
|Training Loss​|4.91​|4.81​|4.78​|4.77​|4.58​|
|Total Time​|1h 05m|1h 54m|1h 44m​|1h 26m​|1h 07m​|
|Cost|$1.12​|$2.71​|$4.68​|$7.9​|$12.1​|
