# Models
The models submodule contains implementations of various algorithms that can be used in addition to external packages to evaluate and develop new natural language processing systems. A description of which algorithms are used in each scenario can be found on [this table](../../README.md#content)

## Summary

The following table summarizes each submodule.

|Submodule|Description|
|---|---|
|[bert](./bert/README.md)| This submodule includes the BERT-based models for sequence classification, token classification, and sequence encoding.|
|[gensen](./gensen/README.md)| This submodule includes a distributed Pytorch implementation based on [Horovod](https://github.com/horovod/horovod) of [learning general purpose distributed sentence representations via large scale multi-task learning](https://arxiv.org/abs/1804.00079) by refactoring https://github.com/Maluuba/gensen|
|[pretrained embeddings](./pretrained_embeddings) | This submodule provides utilities to download and extract pretrained word embeddings trained with Word2Vec, GloVe, fastText methods.|
|[pytorch_modules](./pytorch_modules/README.md)| This submodule provides Pytorch modules like Gated Recurrent Unit with peepholes. |
|[xlnet](./xlnet/README.md)| This submodule includes the XLNet-based model for sequence classification.|
