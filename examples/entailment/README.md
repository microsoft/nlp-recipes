# Natural Language Inference (NLI)  

This folder provides end-to-end examples of building Natural Language Inference (NLI) models. We
demonstrate the best practices of data preprocessing and model building for NLI task and use the
utility scripts in the [utils_nlp](../../utils_nlp) folder to speed up these processes.  
NLI is one of many NLP tasks that require robust compositional sentence understanding, but it's
simpler compared to other tasks like question answering and machine translation.  
 Currently, we focus on fine-tuning pre-trained BERT model. If you are interested in pre-training your own BERT model, you can view the [AzureML-BERT repo](https://github.com/microsoft/AzureML-BERT), which walks through the process in depth.  We plan to continue adding state-of-the-art models as they come up and welcome community contributions.

## Natural Language Inference

Natural Language Inference or Recognizing Textual Entailment (RTE) is the task of classifying
a pair of premise and hypothesis sentences into three classes: contradiction, neutral, and
entailment. For example,  

|Premise|Hypothesis|Label|
|-------|----------|-----|
|A man inspects the uniform of a figure in some East Asian country.|The man is sleeping.|contradiction|
|An older and younger man smiling.|Two men are smiling and laughing at the cats playing on the floor.|neutral|
|A soccer game with multiple males playing.|Some men are playing a sport.|entailment|

## Summary

|Notebook|Environment|Description|Dataset| Language | 
|--------|:-----------:|-------|----------|---------| 
|[entailment_multinli_transformers.ipynb](entailment_multinli_transformers.ipynb)|Local|Fine-tuning of pre-trained BERT model for NLI|[MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)| en | 
|[entailment_xnli_bert_azureml.ipynb](entailment_xnli_bert_azureml.ipynb)|AzureML|**Distributed** fine-tuning of pre-trained BERT model for NLI|[XNLI](https://www.nyu.edu/projects/bowman/xnli/)| en 
