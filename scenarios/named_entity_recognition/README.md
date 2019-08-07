# Named Entity Recognition (NER)

This folder contains examples and best practices, written in Jupyter notebooks, for building Named Entity Recognition models. The models can be used in a wide variety of applications, such as information extraction and filtering. It also plays an important role in other
NLP tasks like question answering and text summarization.

## What is Named Entity Recognition (NER)

Named Entity Recognition (NER) is the task of detecting and classifying
real-world objects mentioned in text. Common named entities include person
names, locations, organizations, etc. The state-of-the art NER methods include
combining Long Short-Term Memory neural network with Conditional Random Field
(LSTM-CRF) and pretrained language models like BERT.

The figure below illustrates how BERT can be fine tuned for NER tasks. The input data is a list of tokens representing a sentence. In the training data, each token has an entity label. After fine tuning, the model predicts an entity label for each token in a given testing sentence.

![Fine-tuned BERT for NER tasks](https://nlpbp.blob.core.windows.net/images/bert_architecture.png)

## Summary

The following summarizes each notebook for NER. Each notebook provides more details and guiding in principles on building state of the art models.

|Notebook|Runs Local|Description|
|---|---|---|
|[Bert](ner_wikigold_bert.ipynb)| Yes| Fine-tune a [pretrained BERT model](https://github.com/huggingface/pytorch-pretrained-BERT) using the [wikigold dataset](https://www.aclweb.org/anthology/W09-3302)  for token classification.|
