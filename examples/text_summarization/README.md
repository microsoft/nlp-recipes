# Text Summarization
This folder contains examples and best practices, written in Jupyter notebooks, for building text Summarization models. We use the
utility scripts in the [utils_nlp](../../utils_nlp) folder to speed up data preprocessing and model building for text Summarization.  

The models can be used in a wide variety of summarization applications, such as abstractive and extractive summarization using transformer-based models. The folder also contains examples for distributed training and inference.

Currently, we focus on fine-tuning pre-trained **BERTSumAbs**, **BertSumExt** and **DistillBERT** models. We plan to continue adding state-of-the-art models as they come up and welcome community contributions.

## What is Text Summarization?
Text Summarization is an unsupervised learning method of a text span that conveys important information of the original text while being significantly shorter. The state-of-the-art methods are based on neural
networks of different architectures as well as pre-trained language models or word embeddings.  

### Extractive summarization
This type of summarization identifies relevant subset sentences from the input text and uses them verbatim while still maintaining the original context of the input text. 

### Abstractive summarization
This type of summarization produces summary by generating entirely new text that conveys important information from the original text.  



## Summary

The following summarizes each notebook for Text Summarization. They are grouped into **Abstractive, Extractive and Evaluation**. The evaluation folder holds a notebook that goes over the evaluation metrics used in the other notebooks.  Each notebook provides more details and guiding in principles on building state of the art models.

|Notebook|Type|Environment|Description|Dataset|
|---|---|---|---|---|
|[Distributed BERTSum for Abstractive Text Summarization on AzureML](./abstractive/abstractive_summarization_bertsum_cnndm_distributed_train.py) |[Abstractive](./abstractive)|Azure ML|A notebook which walks through fine-tuning and evaluating pre-trained BERTSum model for abstractive summarization on a distributed setup with AzureML. |CNN/DailyMail|
|[Abstractive Text Summarization using BertSumAbs](./abstractive/abstractive_summarization_bertsumabs_cnndm.ipynb) |[Abstractive](./abstractive)|Local|A notebook which walks through fine-tuning and evaluating pre-trained BERTSumAbs model for abstractive summarization |CNN/DailyMail|
|[Abstractive Text Summarization using MiniLM](./abstractive/abstractive_summarization_minilm_cnndm.ipynb) |[Abstractive](./abstractive)|Local|A notebook which walks through fine-tuning and evaluating pre-trained MiniLM model for abstractive summarization |CNN/DailyMail|
|[Abstractive Text Summarization using UniLM](./abstractive/abstractive_summarization_unilm_cnndm.ipynb) |[Abstractive](./abstractive)|Local|A notebook which walks through fine-tuning and evaluating pre-trained UniLM model for abstractive summarization |CNN/DailyMail|
|[Distributed BERTSum for Extractive Text Summarization on AzureML](./extractive/extractive_summarization_cnndm_aml_distributed/extractive_summarization_cnndm_aml_distributed.ipynb) |[Extractive](./extractive)|Azure ML|A notebook which walks through fine-tuning and evaluating pre-trained BERTSum model for extractive summarization on a distributed setup with AzureML. |CNN/DailyMail|
|[Extractive Text Summarization using Transformers](./extractive/extractive_summarization_cnndm_transformer.ipynb) |[Extractive](./extractive)|Local|A notebook which walks through fine-tuning and evaluating pre-trained transformers model for extractive summarization |CNN/DailyMail|

