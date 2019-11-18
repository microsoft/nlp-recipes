# Question Answering (QA)

This folder contains examples and best practices, written in Jupyter notebooks, for building
question answering models. These models can be used in a wide variety of applications, such as
search engines, and virtual assistants.


## What is Question Answering?

Question Answering is a classical NLP task which consists of determining the relevant "answer"
(snippet of text out of a provided passage) that answers a user's "question". This task is a subset
of Machine Comprehension, or measuring how well a machine comprehends a passage of text. The
Stanford Question Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/))
leader board displays the state-of-the-art models in this space. Traditional QA models are variants
of Bidirectional Recurrent Neural Networks (BRNN).

## Summary

|Notebook|Environment|Description|Dataset | Language
|---|---|---|---|----|
|[Deployed QA System in Under 20 minutes](question_answering_system_bidaf_quickstart.ipynb)|Azure Container Instances| Learn how to deploy a QA system in under 20 minutes using Azure Container Instances (ACI) and a popular AllenNLP pre-trained model called BiDAF.|[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)| English | 
|[BiDAF Deep Dive](bidaf_aml_deep_dive.ipynb)|Azure ML| Learn about the architecture of the BiDAF model and how to train it from scratch using the AllenNLP library on the AzureML platform.|[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) | English |
|[Pretrained BERT SQuAD Deep Dive](pretrained-BERT-SQuAD-deep-dive-aml.ipynb)|Azure ML| Learn about the mechanism of the BERT model in an end to end pipeline on the AzureML platform and how to fine tune it from scratch using the distributed training with Horovod. Show the improvement on the model performance using hyper-parameter tuning|[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)| English |

