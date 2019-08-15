### Models
The models submodule contains implementations of various algorithms that can be used in addition to external packages to evaluate and develop new natural language processing systems. A description of which algorithms are used in each scenario can be found on [this table](../../README.md#content)

This includes:
* [BERT](./bert/common.py)
* [GenSen](./gensen/gensen.py)
* Pretrained embeddings ([Word2Vec](./pretrained_embeddings/word2vec.py),
[fastText](./pretrained_embeddings/fasttext.py),
[GloVe](./pretrained_embeddings/glove.py))
* Pytorch's conditional Gated Recurrent Unit ([GRU](./pytorch_modules/conditional_gru.py))