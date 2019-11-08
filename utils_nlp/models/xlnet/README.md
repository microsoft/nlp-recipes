# XLNet-based Classes

This folder contains utility functions and classes based on the implementation of [Transformers](https://github.com/huggingface/transformers). 

## Summary

The following table summarizes each Python script.

|Script|Description|
|---|---|
|[common.py](common.py)| This script includes <ul><li>the languages supported by XLNet-based classes</li><li> tokenization for text classification</li> <li>utilities to load data, etc.</li></ul>|
|[sequence_classification.py](sequence_classification.py)| An implementation of sequence classification based on fine-turning XLNet. It is commonly used for text classification. The module includes logging functionality using MLFlow.|
|[utils.py](utils.py)| This script includes a function to visualize a confusion matrix.|
