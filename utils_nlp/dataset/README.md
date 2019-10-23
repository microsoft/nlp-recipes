## [Dataset](.)
This submodule includes helper functions for downloading datasets and formatting them appropriately as well as utilities for splitting data for training / testing.

## Data Loading
There are dataloaders for several datasets. For example, the snli module will allow you to load a dataframe in pandas from the SNLI dataset, with the option to set the number of rows to load in order to test algorithms and evaluate performance benchmarks.
Most datasets may be split into `train`, `dev`, and `test`, for example:

```python
from utils_nlp.dataset.snli import load_pandas_df

df = load_pandas_df(DATA_FOLDER, file_split ="train", nrows = 1000)
```
## Dataset List
|Dataset|Dataloader script|
|-------|-----------------|
|[Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398)|[msrpc.py](./msrpc.py)|
|[The Multi-Genre NLI (MultiNLI) Corpus](https://www.nyu.edu/projects/bowman/multinli/)|[multinli.py](./multinli.py)|
|[The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)|[snli.py](./snli.py)|
|[Wikigold NER](https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold/CONLL-format/data)|[wikigold.py](./wikigold.py)|
|[The Cross-Lingual NLI (XNLI) Corpus](https://www.nyu.edu/projects/bowman/xnli/)|[xnli.py](./xnli.py)|
|[The STSbenchmark dataset](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)|[stsbenchmark.py](./stsbenchmark.py)|
|[The Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/)|[squad.py](./squad.py)|

## Dataset References
Please see [Dataset References](../../DatasetReferences.md) for notice and information regarding datasets used.
