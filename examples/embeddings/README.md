# Word Embedding

This folder contains examples and best practices, written in Jupyter notebooks, for training word embedding on custom data from scratch.   
There are
three typical ways for training word embedding:
[Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf),
[GloVe](https://nlp.stanford.edu/pubs/glove.pdf), and [fastText](https://arxiv.org/abs/1607.01759).
All of the three methods provide pretrained models ([pretrained model with
Word2Vec](https://code.google.com/archive/p/word2vec/), [pretrained model with
Glove](https://github.com/stanfordnlp/GloVe), [pretrained model with
fastText](https://fasttext.cc/docs/en/crawl-vectors.html)).   
These pretrained models are trained with
general corpus like Wikipedia data, Common Crawl data, etc., and may not serve well for situations
where you have a domain-specific language learning problem or there is no pretrained model for the
language you need to work with.  In this folder, we provide examples of how to apply each of the
three methods to train your own word embeddings.  

# What is Word Embedding?

Word embedding is a technique to map words or phrases from a vocabulary to vectors or real numbers.
The learned vector representations of words capture  syntactic and semantic word relationships and
therefore can be very useful for  tasks like sentence similary, text classifcation, etc.


## Summary


|Notebook|Environment|Description|Dataset| Language | 
|---|---|---|---|---|
|[Developing Word Embeddings](embedding_trainer.ipynb)|Local| A notebook shows how to learn word representation with Word2Vec, fastText and Glove|[STS Benchmark dataset](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#STS_benchmark_dataset_and_companion_dataset) | en |
