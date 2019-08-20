# NLP Best Practices

This repository contains examples and best practices for building natural language processing (NLP) systems, provided as [Jupyter notebooks](scenarios) and [utility functions](utils_nlp). The focus of the repository is on state-of-the-art methods and common scenarios that are popular among researchers and practitioners working on problems involving text and language.

![](https://nlpbp.blob.core.windows.net/images/cognitive_services.PNG)
## Overview

The goal of this repository is to build a comprehensive set of tools and examples that leverage recent advances in NLP algorithms, neural architectures, and distributed machine learning systems.
The content is based on our past and potential future engagements with customers as well as collaboration with partners, researchers, and the open source community.

We’re hoping that the tools would significantly reduce the time from a business problem, or a research idea, to full implementation of a system. In addition, the example notebooks would serve as guidelines and showcase best practices and usage of the tools.

In an era of transfer learning, transformers, and deep architectures, we believe that pretrained models provide a unified solution to many real-world problems and allow handling different tasks and languages easily. We will, therefore, prioritize such models, as they achieve state-of-the-art results on several NLP benchmarks and can be used in a number of applications ranging from simple text classification to sophisticated intelligent chat bots.

> [*GLUE Leaderboard*](https://gluebenchmark.com/leaderboard)  
> [*SQuAD Leaderbord*](https://rajpurkar.github.io/SQuAD-explorer/)

## Content

The following is a summary of the scenarios covered in the repository. Each scenario is demonstrated in one or more Jupyter notebook examples that make use of the core code base of models and utilities.

| Scenario                 | Applications                                 |  Models |
|---| ------------------------ | ------------------- |
|[Text Classification](scenarios/text_classification)      |Topic Classification|BERT|
|[Named Entity Recognition](scenarios/named_entity_recognition) |Wikipedia NER                                              |BERT|
|[Entailment](scenarios/entailment)|MultiNLI Natural Language Inference|BERT|
|[Question Answering](scenarios/question_answering) |SQuAD                                              | BiDAF, BERT|
|[Sentence Similarity](scenarios/sentence_similarity)      |STS Benchmark                         |Representation: TF-IDF, Word Embeddings, Doc Embeddings<br>Metrics: Cosine Similarity, Word Mover's Distance|
|[Embeddings](scenarios/embeddings)| Custom Embeddings Training|Word2Vec<br>fastText<br>GloVe|
| [Annotation](scenarios/annotation) | Text annotation | Tutorial |



## Getting Started
To get started, navigate to the [Setup Guide](SETUP.md), where you'll find instructions on how to setup your environment and dependencies.

## Contributing
This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


## Build Status
| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/cpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=50&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/cpu_integration_tests_linux?branchName=staging)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=50&branchName=staging) |
| **Linux GPU** | master | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/gpu_integration_tests_linux?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=51&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/gpu_integration_tests_linux?branchName=staging)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=51&branchName=staging) |
