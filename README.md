# NLP Best Practices

In recent years, Natural Language Processing has seen quick growth in quality and usability, and this has helped to drive business adoption of Artificial Intelligence solutions. In the last few years, researchers have been applying newer deep learning methods to natural language processing. Data Scientists started moving from traditional methods to state-of-the-art DNN algorithms which allow them to use language models pretrained on large text corpora.

This repository contains examples and best practices for building natural language processing (NLP) systems, provided as [Jupyter notebooks](examples) and [utility functions](utils_nlp). The focus of the repository is on state-of-the-art methods and common scenarios that are popular among researchers and practitioners working on problems involving text and language.

## Overview

The goal of this repository is to build a comprehensive set of tools and examples that leverage recent advances in NLP algorithms, neural architectures, and distributed machine learning systems.
The content is based on our past and potential future engagements with customers as well as collaboration with partners, researchers, and the open source community.

We’re hoping that the tools would significantly reduce the “time to market” by simplifying the experience from defining the business problem to development of solution by orders of magnitude. In addition, the example notebooks would serve as guidelines and showcase best practices and usage of the tools in a wide variety of languages.

In an era of transfer learning, transformers, and deep architectures, we believe that pretrained models provide a unified solution to many real-world problems and allow handling different tasks and languages easily. We will, therefore, prioritize such models, as they achieve state-of-the-art results on several NLP benchmarks like [*GLUE*](https://gluebenchmark.com/leaderboard) and [*SQuAD*](https://rajpurkar.github.io/SQuAD-explorer/) leaderboard. The models can be used in a number of applications ranging from simple text classification to sophisticated intelligent chat bots.

>   
> 

## Content
The following is a summary of the commonly used NLP scenarios covered in the repository. Each scenario is demonstrated in one or more [Jupyter notebook examples](examples) that make use of the core code base of models and repository utilities.

| Scenario                              |  Models | Description|
|-------------------------|  ------------------- |-------|
|Text Classification                     |BERT| Text classification is a supervised learning method of learning and predicting the category or the class of a document given its text content. | 
|Named Entity Recognition                |BERT| Named Entity Recognition (NER) is the task of classifying words or key phrases of a text into predefined entities of interest. |
|Entailment                              |BERT| Textual entailment is a binary relation between two natural-language texts (called ‘text’ and ‘hypothesis’), where readers of the ‘text’ would agree the ‘hypothesis’ is most likely true. |
|Question Answering                      |BiDAF| Question Answering (QA) is the task of retrieving or generating a valid answer for a given natural language query. |
|Sentence Similarity                     |Representation: TF-IDF, Word Embeddings, Doc Embeddings<br>Metrics: Cosine Similarity, Word Mover's Distance| Sentence similarity is the process of computing a similarity score given a pair of text documents. |
|Embeddings| Word2Vec<br>fastText<br>GloVe| An embedding is a low dimensionality representation of the text that will be analyzed.


## Getting Started
While solving NLP problems, its always good to start with [Language-based Cognitive Services](https://azure.microsoft.com/en-us/services/cognitive-services/directory/lang/). When the needs are beyond the bounds of Cognitive Services, you can try custom Machine Learning and this is where the repository can be very useful. To get started, navigate to the [Setup Guide](SETUP.md), where you'll find instructions on how to setup your environment and dependencies.

## Contributing
We hope that the open source community would contribute to the content and bring in the latest SOTA algorithm. This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


## Build Status
| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/cpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=50&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/cpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=50&branchName=staging) |
| **Linux GPU** | master | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/gpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=51&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/gpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=51&branchName=master) |
