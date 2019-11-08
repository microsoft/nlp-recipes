# BERT-based Classes

This folder contains utility functions and classes based on the implementation of [Transformers](https://github.com/huggingface/transformers). 

## Summary

The following table summarizes each Python scripts.

|Script|Description|
|---|---|
|[common.py](common.py)| This script includes <ul><li>the languages supported by BERT-based classes</li><li> tokenization for text classification, name entity recognition, and encoding</li> <li>utilities to load data, etc.</li></ul>|
|[sequence_classification.py](sequence_classification.py)| An implementation of sequence classification based on fine-turning BERT. It is commonly used for text classification.|
|[sequence_classification_distributed.py](sequence_classification_distributed.py) | A distributed implementation of sequence classification with method based on fine-turning BERT. [Horovod](https://github.com/horovod/horovod) is the underlying distributed training framework.|
|[sequence_encoding.py](sequence_encoding.py)| An implementation of sequence encoding based on BERT. Both pretrained and fine-tuned BERT models can be used. The hidden states from the loaded BERT model for the input sequence are used in the computation of the encoding. It provides mean, max and class pooling stragegies. It is commonly used in upstream tasks for sentence similarity. |
|[token_classification.py](token_classification.py) |  An implementation of token classification based on fine-turning BERT. It is commonly used for name entity recognition. |
