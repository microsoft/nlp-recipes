# Named Entity Recognition (NER)

This folder contains examples and best practices, written in Jupyter notebooks, for building Named
Entity Recognition models. We use the
utility scripts in the [utils_nlp](../../utils_nlp) folder to speed up data preprocessing and model building for NER.  
The models can be used in a wide variety of applications, such as
information extraction and filtering. It also plays an important role in other NLP tasks like
question answering and text summarization.  
Currently, we focus on fine-tuning pre-trained BERT
model. We plan to continue adding state-of-the-art models as they come up and welcome community
contributions.

## What is Named Entity Recognition (NER)

Named Entity Recognition (NER) is the task of detecting and classifying real-world objects mentioned
in text. Common named entities include person names, locations, organizations, etc. The
[state-of-the art](https://paperswithcode.com/task/named-entity-recognition-ner) NER methods include
combining Long Short-Term Memory neural network with Conditional Random Field (LSTM-CRF) and
pretrained language models like BERT.

NER usually involves assigning an entity label to each word in a sentence as shown in the figure below.   
<p align="center">
  <img src="https://nlpbp.blob.core.windows.net/images/ner.PNG" alt=" Fine-tuned BERT for NER tasks"/>
</p>

* O:  Not an entity
* I-LOC: Location
* I-ORG: Organization
* I-PER: Person

There are a few standard labeling schemes and you can find the details
[here](http://cs229.stanford.edu/proj2005/KrishnanGanapathy-NamedEntityRecognition.pdf). The data
can also be labeled with custom entities as required by the use case.

## Summary

|Notebook|Environment|Description|Dataset|Language| 
|---|:---:|---|---|---|
|[BERT](ner_wikigold_transformer.ipynb)|Local| Fine-tune a pretrained BERT model for token classification.|[wikigold](https://www.aclweb.org/anthology/W09-3302)| English | 
