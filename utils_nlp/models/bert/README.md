# BERT-based Classes

This folder contains utility functions and classes based on the implementation of [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers). 

## Summary

The following table summarizes each Python scripts.

|Script|Description|
|---|---|
|[common.py](common.py)| This script includes <ul><li>the languages supported by BERT-based classes</li><li> tokenization for text classification and name entity recognition, and encoding</li> <li>utilities to load data, etc.</li></ul>|
|[sequence_classification.py](sequence_classification.py)| An implemention of sequence classification with method of fine-turning BERT. It comminly used for text classification.|
|[sequence_classification_distributed.py](sequence_classification_distributed.py) | An distributed implemention of sequence classification with method of fine-turning BERT. [Horovod](https://github.com/horovod/horovod) is the underlying distributed training framework.|
|[sequence_encoding.py](sequence_encoding.py)| An implemention of sequence encoding based on BERT. The hidden states from the pretrained model for the input sequenced are used in the computation of the encoding. It provides Mean, max and class pooling stragegies. It's commonly used in upstream tasks for sentence similarity. |
|[token_classification.py](token_classification.py) |  An implemention of token classification with method of fine-turning BERT. It's commonly used for name entity recognition. |
