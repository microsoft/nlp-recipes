# Natural Language Inference (NLI)  

This folder provides end-to-end examples of building Natural Language Inference (NLI) models. We
demonstrate the typical workflow of data preprocessing and model building for NLI task and use the
utility functions in the *utils_nlp* folder to speed up these processes.

## Natural Language Inference

Natural Language Inference or Recognizing Textual Entailment (RTE) is the task of classifying
a pair of premise and hypothesis sentences into three classes: contradiction, neutral, and
entailment. For example,  

|Premise|Hypothesis|Label|
|-------|----------|-----|
|A man inspects the uniform of a figure in some East Asian country.|The man is sleeping.|contradiction|
|An older and younger man smiling.|Two men are smiling and laughing at the cats playing on the floor.|neutral|
|A soccer game with multiple males playing.|Some men are playing a sport.|entailment|

NLI is one of many NLP tasks that require robust compositional sentence understanding, but it's
simpler compared to other tasks like question answering and machine translation.

Currently, we focus on fine-tuning pre-trained BERT models for NLI and use the utility classes and
functions under [utils_nlp/models/bert](../../utils_nlp/models/bert/). We plan to continue adding
new state-of-the-art models.

## Folder Content
The following notebooks are included in this folder

|Notebook|Description|Dataset|Runs Local|
|--------|-----------|-------|:----------:|
|entailment_multinli_bert.ipynb|Fine-tuning of pre-trained BERT model for NLI|[MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)|Yes|
|entailment_xnli_bert_azureml.ipynb|**Distributed** fine-tuning of pre-trained BERT model for NLI|[XNLI](https://www.nyu.edu/projects/bowman/xnli/)|Yes
