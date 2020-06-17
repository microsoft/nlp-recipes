<img src="NLP-Logo.png" align="right" alt="" width="300"/>


# NLP Best Practices

In recent years, natural language processing (NLP) has seen quick growth in quality and usability, and this has helped to drive business adoption of artificial intelligence (AI) solutions. In the last few years, researchers have been applying newer deep learning methods to NLP. Data scientists started moving from traditional methods to state-of-the-art (SOTA) deep neural network (DNN) algorithms which use language models pretrained on large text corpora.

This repository contains examples and best practices for building NLP systems, provided as [Jupyter notebooks](examples) and [utility functions](utils_nlp). The focus of the repository is on state-of-the-art methods and common scenarios that are popular among researchers and practitioners working on problems involving text and language.

## Overview

The goal of this repository is to build a comprehensive set of tools and examples that leverage recent advances in NLP algorithms, neural architectures, and distributed machine learning systems.
The content is based on our past and potential future engagements with customers as well as collaboration with partners, researchers, and the open source community.

We hope that the tools can significantly reduce the “time to market” by simplifying the experience from defining the business problem to development of solution by orders of magnitude. In addition, the example notebooks would serve as guidelines and showcase best practices and usage of the tools in a wide variety of languages.

In an era of transfer learning, transformers, and deep architectures, we believe that pretrained models provide a unified solution to many real-world problems and allow handling different tasks and languages easily. We will, therefore, prioritize such models, as they achieve state-of-the-art results on several NLP benchmarks like [*GLUE*](https://gluebenchmark.com/leaderboard) and [*SQuAD*](https://rajpurkar.github.io/SQuAD-explorer/) leaderboards. The models can be used in a number of applications ranging from simple text classification to sophisticated intelligent chat bots.

Note that for certain kind of NLP problems, you may not need to build your own models. Instead, pre-built or easily customizable solutions exist which do not require any custom coding or machine learning expertise. We strongly recommend evaluating if these can sufficiently solve your problem. If these solutions are not applicable, or the accuracy of these solutions is not sufficient, then resorting to more complex and time-consuming custom approaches may be necessary. The following cognitive services offer simple solutions to address common NLP tasks:
<br><br><b>[Text Analytics](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/) </b> are a set of pre-trained REST APIs which can be called for Sentiment Analysis, Key phrase extraction, Language detection and Named Entity Detection and more. These APIs work out of the box and require minimal expertise in machine learning, but have limited customization capabilities.
<br><br><b>[QnA Maker](https://azure.microsoft.com/en-us/services/cognitive-services/qna-maker/) </b>is a cloud-based API service that lets you create a conversational question-and-answer layer over your existing data. Use it to build a knowledge base by extracting questions and answers from your semi-structured content, including FAQs, manuals, and documents.
<br><br><b>[Language Understanding](https://azure.microsoft.com/en-us/services/cognitive-services/language-understanding-intelligent-service/)</b> is a SaaS service to train and deploy a model as a REST API given a user-provided training set. You could do Intent Classification as well as Named Entity Extraction by performing simple steps of providing example utterances and labelling them. It supports Active Learning, so your model always keeps learning and improving.

## Target Audience
For this repository our target audience includes data scientists and machine learning engineers with varying levels of NLP knowledge as our content is source-only and targets custom machine learning modelling. The utilities and examples provided are intended to be solution accelerators for real-world NLP problems.

## Focus Areas
The repository aims to expand NLP capabilities along three separate dimensions

### Scenarios
We aim to have end-to-end examples of common tasks and scenarios such as text classification, named entity recognition etc.

### Algorithms
We aim to support multiple models for each of the supported scenarios. Currently, transformer-based models are supported across most scenarios. We have been working on integrating the [transformers package](https://github.com/huggingface/transformers) from [Hugging Face](https://huggingface.co/) which allows users to easily load pretrained models and fine-tune them for different tasks.

### Languages
We strongly subscribe to the multi-language principles laid down by ["Emily Bender"](http://faculty.washington.edu/ebender/papers/Bender-SDSS-2019.pdf)
* "Natural language is not a synonym for English"
* "English isn't generic for language, despite what NLP papers might lead you to believe"
* "Always name the language you are working on" ([Bender rule](https://www.aclweb.org/anthology/Q18-1041/))

The repository aims to support non-English languages  across all the scenarios. Pre-trained models used in the repository such as BERT, FastText support 100+ languages out of the box. Our goal is to provide end-to-end examples in as many languages as possible. We encourage community contributions in this area.



## Content
The following is a summary of the commonly used NLP scenarios covered in the repository. Each scenario is demonstrated in one or more [Jupyter notebook examples](examples) that make use of the core code base of models and repository utilities.

| Scenario                              |  Models | Description|Languages|
|-------------------------|  ------------------- |-------|---|
|Text Classification                     |BERT, DistillBERT, XLNet, RoBERTa, ALBERT, XLM| Text classification is a supervised learning method of learning and predicting the category or the class of a document given its text content. |English, Chinese, Hindi, Arabic, German, French, Japanese, Spanish, Dutch|
|Named Entity Recognition                |BERT| Named entity recognition (NER) is the task of classifying words or key phrases of a text into predefined entities of interest. |English|
|Text Summarization|BERTSumExt <br> BERTSumAbs <br> UniLM (s2s-ft) <br> MiniLM |Text summarization is a language generation task of summarizing the input text into a shorter paragraph of text.|English
|Entailment                              |BERT, XLNet, RoBERTa| Textual entailment is the task of classifying the binary relation between two natural-language texts,  *text* and *hypothesis*, to determine if the *text* agrees with the *hypothesis* or not. |English|
|Question Answering                      |BiDAF, BERT, XLNet| Question answering (QA) is the task of retrieving or generating a valid answer for a given query in natural language, provided with a passage related to the query. |English|
|Sentence Similarity                     |BERT, GenSen| Sentence similarity is the process of computing a similarity score given a pair of text documents. |English|
|Embeddings| Word2Vec<br>fastText<br>GloVe| Embedding is the process of converting a word or a piece of text to a continuous vector space of real number, usually, in low dimension.|English|
|Sentiment Analysis| Dependency Parser <br>GloVe| Provides an example of train and use Aspect Based Sentiment Analysis with Azure ML and [Intel NLP Architect](http://nlp_architect.nervanasys.com/absa.html) .|English|
## Getting Started
While solving NLP problems, it is always good to start with the prebuilt [Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services/directory/lang/). When the needs are beyond the bounds of the prebuilt cognitive service and when you want to search for custom machine learning methods,  you will find this repository  very useful. To get started, navigate to the [Setup Guide](SETUP.md), which lists instructions on how to setup your environment and dependencies.


## Azure Machine Learning Service
[Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) is a cloud service used to train, deploy, automate, and manage machine learning models, all at the broad scale that the cloud provides. AzureML is presented in notebooks across different scenarios to enhance the efficiency of developing Natural Language systems at scale and for various AI model development related tasks like:
  * [**Accessing Datastores**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-access-data) to easily read and write your data in Azure storage services such as blob storage or file share.
  * Scaling up and out on [**Azure Machine Learning Compute**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets#amlcompute).
  * [**Automated Machine Learning**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train) which builds high quality machine learning models by automating model and hyperparameter selection. AutoML explores BERT, BiLSTM, bag-of-words, and word embeddings on the user's dataset to handle text columns.
  * [**Tracking experiments and monitoring metrics**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-track-experiments) to enhance the model creation process.
  * [**Distributed Training**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-ml-models#distributed-training-and-custom-docker-images)
  * [**Hyperparameter tuning**](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters)
  * Deploying the trained machine learning model as a web service to [**Azure Container Instance**](https://azure.microsoft.com/en-us/services/container-instances/) for deveopment and test,  or for low scale, CPU-based workloads.
  * Deploying the trained machine learning model as a web service to [**Azure Kubernetes Service**](https://azure.microsoft.com/en-us/services/kubernetes-service/) for high-scale production deployments and provides autoscaling, and fast response times.

To successfully run these notebooks, you will need an [**Azure subscription**](https://azure.microsoft.com/en-us/) or can [**try Azure for free**](https://azure.microsoft.com/en-us/free/). There may be other Azure services or products used in the notebooks. Introduction and/or reference of those will be provided in the notebooks themselves.

## Contributing
We hope that the open source community would contribute to the content and bring in the latest SOTA algorithm. This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).

## Blog Posts

- [Bootstrap Your Text Summarization Solution with the Latest Release from NLP-Recipes](https://techcommunity.microsoft.com/t5/ai-customer-engineering-team/bootstrap-your-text-summarization-solution-with-the-latest/ba-p/1268809)

- [Text Annotation made easy with Doccano](https://techcommunity.microsoft.com/t5/ai-customer-engineering-team/text-annotation-made-easy-with-doccano/ba-p/1242612)

- [Jumpstart Analyzing your Hindi Text Data using the NLP Repository](https://techcommunity.microsoft.com/t5/ai-customer-engineering-team/jumpstart-analyzing-your-hindi-text-data-using-the-nlp/ba-p/1087851)

- [Speeding up the Development of Natural Language Processing Solutions with Azure Machine Learning](https://techcommunity.microsoft.com/t5/ai-customer-engineering-team/speeding-up-the-development-of-natural-language-processing/ba-p/1042577)

## References
The following is a list of related repositories that we like and think are useful for NLP tasks.

|Repository|Description|
|---|---|
|[Transformers](https://github.com/huggingface/transformers)|A great PyTorch library from Hugging Face with implementations of popular transformer-based models. We've been using their package extensively in this repo and greatly appreciate their effort.|
|[Azure Machine Learning Notebooks](https://github.com/Azure/MachineLearningNotebooks/)|ML and deep learning examples with Azure Machine Learning.|
|[AzureML-BERT](https://github.com/Microsoft/AzureML-BERT)|End-to-end recipes for pre-training and fine-tuning BERT using Azure Machine Learning service.|
|[MASS](https://github.com/microsoft/MASS)|MASS: Masked Sequence to Sequence Pre-training for Language Generation.|
|[MT-DNN](https://github.com/microsoft/mt-dnn)|Multi-Task Deep Neural Networks for Natural Language Understanding.|
|[UniLM](https://github.com/microsoft/unilm)|Unified Language Model Pre-training.|
|[DialoGPT](https://github.com/microsoft/DialoGPT)|DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation|


## Build Status
| Build | Branch | Status |
| --- | --- | --- |
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/cpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=50&branchName=master) |
| **Linux CPU** | staging | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/cpu_integration_tests_linux?branchName=staging)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=50&branchName=staging) |
| **Linux GPU** | master | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/gpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=51&branchName=master) |
| **Linux GPU** | staging | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/gpu_integration_tests_linux?branchName=staging)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=51&branchName=staging) |
