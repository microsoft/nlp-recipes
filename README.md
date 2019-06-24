
| Branch | Status                                                                                                                                                                                                      |     | Branch  | Status                                                                                                                                                                                                         |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| master | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/unit-test-master?branchName=master)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=22&branchName=master) |     | staging | [![Build Status](https://dev.azure.com/best-practices/nlp/_apis/build/status/unit-test-staging?branchName=staging)](https://dev.azure.com/best-practices/nlp/_build/latest?definitionId=21&branchName=staging) |


# NLP Best Practices

This repository contains examples and best practices for building NLP systems, provided as Jupyter notebooks and utility functions. The focus of the repository is on state-of-the-art methods and common scenarios that are popular among researchers and practitioners working on problems involving text and language.

The following section includes a list of the available scenarios. Each scenario is demonstrated in one or more Jupyter notebook examples that make use of the core code base of models and utilities.


## Scenarios


| Scenario                 | Applications                                 | Languages | Models |
|---| ------------------------ | -------------------------------------------- | ------------------- |
|[Text Classification](scenarios/text_classification)      |Topic Classification|en, zh, ar|BERT|
|[Named Entity Recognition](scenarios/named_entity_recognition) |Wikipedia NER                                              | en, zh  |BERT|
|[Sentence Similarity](scenarios/sentence_similarity)      |STS Benchmark                         |en|Representation: TF-IDF, Word Embeddings, Doc Embeddings<br>Metrics: Cosine Similarity, Word Mover's Distance|
|[Embeddings](scenarios/embeddings)| Custom Embeddings Training|en|Word2Vec<br>fastText<br>GloVe|


## Planning
All feature planning is done via projects, milestones, and issues in this repository.

## Getting Started
To get started, navigate to the [Setup Guide](SETUP.md), where you'll find instructions on how to setup your environment and dependencies.

## Contributing
This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).
