# Text Classification
This folder contains examples and best practices, written in Jupyter notebooks, for building text classification models. We use the
utility scripts in the [utils_nlp](../../utils_nlp) folder to speed up data preprocessing and model building for text classification.  
The models can be used in a wide variety of applications, such as
sentiment analysis, document indexing in digital libraries, hate speech detection, and general-purpose categorization in medical, academic, legal, and many other domains. 
Currently, we focus on fine-tuning pre-trained BERT
model. We plan to continue adding state-of-the-art models as they come up and welcome community
contributions.

## What is Text Classification?
Text classification is a supervised learning method of learning and predicting the category or the
class of a document given its text content. The state-of-the-art methods are based on neural
networks of different architectures as well as pre-trained language models or word embeddings.


## Summary

The following summarizes each notebook for Text Classification. Each notebook provides more details and guiding in principles on building state of the art models.

|Notebook|Environment|Description|Dataset|
|---|---|---|---|
|[BERT for text classification with MNLI](tc_mnli_bert.ipynb)|Local| A notebook which walks through fine-tuning and evaluating a pre-trained BERT model on a subset of the MultiNLI dataset|[MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)|
|[BERT for text classification on AzureML](tc_bert_azureml.ipynb) |Azure ML|A notebook which walks through fine-tuning and evaluating pre-trained BERT model on a distributed setup with AzureML. |[MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)|
