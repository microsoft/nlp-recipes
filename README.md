# NLP Best Practices

In recent years, natural language processing (NLP) has seen quick growth in quality and usability, and this has helped to drive business adoption of artificial intelligence (AI) solutions. In the last few years, researchers have been applying newer deep learning methods to NLP. Data scientists started moving from traditional methods to state-of-the-art (SOTA) deep neural network (DNN) algorithms which use language models pretrained on large text corpora.

This repository contains examples and best practices for building NLP systems, provided as [Jupyter notebooks](examples) and [utility functions](utils_nlp). The focus of the repository is on state-of-the-art methods and common scenarios that are popular among researchers and practitioners working on problems involving text and language.

## Overview

The goal of this repository is to build a comprehensive set of tools and examples that leverage recent advances in NLP algorithms, neural architectures, and distributed machine learning systems.
The content is based on our past and potential future engagements with customers as well as collaboration with partners, researchers, and the open source community.

We hope that the tools can significantly reduce the “time to market” by simplifying the experience from defining the business problem to development of solution by orders of magnitude. In addition, the example notebooks would serve as guidelines and showcase best practices and usage of the tools in a wide variety of languages.

In an era of transfer learning, transformers, and deep architectures, we believe that pretrained models provide a unified solution to many real-world problems and allow handling different tasks and languages easily. We will, therefore, prioritize such models, as they achieve state-of-the-art results on several NLP benchmarks like [*GLUE*](https://gluebenchmark.com/leaderboard) and [*SQuAD*](https://rajpurkar.github.io/SQuAD-explorer/) leaderboards. The models can be used in a number of applications ranging from simple text classification to sophisticated intelligent chat bots.

>   
>

## Content
The following is a summary of the commonly used NLP scenarios covered in the repository. Each scenario is demonstrated in one or more [Jupyter notebook examples](examples) that make use of the core code base of models and repository utilities.

| Scenario                              |  Models | Description|
|-------------------------|  ------------------- |-------|
|Text Classification                     |BERT| Text classification is a supervised learning method of learning and predicting the category or the class of a document given its text content. |
|Named Entity Recognition                |BERT| Named entity recognition (NER) is the task of classifying words or key phrases of a text into predefined entities of interest. |
|Entailment                              |BERT| Textual entailment is the task of classifying the binary relation between two natural-language texts,  ‘text’ and ‘hypothesis’,  to determine if the `text' agrees with the `hypothesis` or not. |
|Question Answering                      |BiDAF <br> BERT| Question answering (QA) is the task of retrieving or generating a valid answer for a given query in natural language, provided with a passage related to the query. |
|Sentence Similarity                     |Representation: TF-IDF, Word Embeddings, Doc Embeddings<br>Metrics: Cosine Similarity, Word Mover's Distance<br>Models: BERT, GenSen| Sentence similarity is the process of computing a similarity score given a pair of text documents. |
|Embeddings| Word2Vec<br>fastText<br>GloVe| Embedding is the process of converting a word or a piece of text to a continuous vector space of real number, usually, in low dimension.


## Getting Started
While solving NLP problems, it is always good to start with the prebuilt [Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services/directory/lang/). When the needs are beyond the bounds of the prebuilt cognitive service and when you want to search for custom machine learning methods,  you will find this repository  very useful. To get started, navigate to the [Setup Guide](SETUP.md), which lists instructions on how to setup your environment and dependencies.

## Azure Machine Learning service
[Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) is a cloud service used to train, deploy, automate, and manage machine learning models, all at the broad scale that the cloud provides. AzureML is presented in notebooks across different scenarios to enhance the efficiency of developing Natural Language systems at scale and for various AI model development related tasks like:
  * [**Accessing Datastores**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-access-data) to easily read and write your data in Azure storage services such as blob storage or file share.
  * Scaling up and out on [**Azure Machine Learning Compute**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets#amlcompute).
  * [**Automated Machine Learning**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train) which builds high quality machine learning models by automating model and hyperparameter selection.
  * [**Tracking experiments and monitoring metrics**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-track-experiments) to enhance the model creation process.
  * [**Distributed Training**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-ml-models#distributed-training-and-custom-docker-images)
  * [**Hyperparameter tuning**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters)
  * Deploying the trained machine learning model as a web service to [**Azure Container Instance**](https://azure.microsoft.com/en-us/services/container-instances/) for deveopment and test,  or for low scale, CPU-based workloads.
  * Deploying the trained machine learning model as a web service to [**Azure Kubernetes Service**](https://azure.microsoft.com/en-us/services/kubernetes-service/) for high-scale production deployments and provides autoscaling, and fast response times.

To successfully run these notebooks, you will need an [**Azure subscription**](https://azure.microsoft.com/en-us/) or can [**try Azure for free**](https://azure.microsoft.com/en-us/free/). There may be other Azure services or products used in the notebooks. Introduction and/or reference of those will be provided in the notebooks themselves.

## Contributing
We hope that the open source community would contribute to the content and bring in the latest SOTA algorithm. This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


## Build Status
| Build Type | Branch | Status |  | Branch | Status |
| --- | --- | --- | --- | --- | --- |
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/cpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=50&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/cpu_integration_tests_linux?branchName=staging)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=50&branchName=staging) |
| **Linux GPU** | master | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/gpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=51&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/gpu_integration_tests_linux?branchName=staging)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=51&branchName=staging) |
