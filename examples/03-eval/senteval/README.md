# SentEval: evaluation toolkit for sentence embeddings

SentEval is a library for evaluating the quality of sentence embeddings. We assess their generalization power by using them as features on a broad and diverse set of "transfer" tasks. **SentEval currently includes 17 downstream tasks**. We also include a suite of **10 probing tasks** which evaluate what linguistic properties are encoded in sentence embeddings. Our goal is to ease the study and the development of general-purpose fixed-size sentence representations.


**(04/22) SentEval new tasks: Added probing tasks for evaluating what linguistic properties are encoded in sentence embeddings**

**(10/04) SentEval example scripts for three sentence encoders: [SkipThought-LN](https://github.com/ryankiros/layer-norm#skip-thoughts)/[GenSen](https://github.com/Maluuba/gensen)/[Google-USE](https://tfhub.dev/google/universal-sentence-encoder/1)**

## Dependencies

This code is written in python. The dependencies are:

* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
* [Pytorch](http://pytorch.org/)>=0.4
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0

## Transfer tasks

### Downstream tasks
SentEval allows you to evaluate your sentence embeddings as features for the following *downstream* tasks:

| Task     	| Type                         	| #train 	| #test 	| needs_train 	| set_classifier |
|----------	|------------------------------	|-----------:|----------:|:-----------:|:----------:|
| [MR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm)       	| movie review                 	| 11k     	| 11k    	| 1 | 1 |
| [CR](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm)       	| product review               	| 4k      	| 4k     	| 1 | 1 |
| [SUBJ](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm)     	| subjectivity status          	| 10k     	| 10k    	| 1 | 1 |
| [MPQA](https://nlp.stanford.edu/~sidaw/home/projects:nbsvm)     	| opinion-polarity  | 11k     	| 11k    	| 1 | 1 |
| [SST](https://nlp.stanford.edu/sentiment/index.html)      	| binary sentiment analysis  	| 67k     	| 1.8k   	| 1 | 1 |
| **[SST](https://nlp.stanford.edu/sentiment/index.html)**      	| **fine-grained sentiment analysis**  	| 8.5k     	| 2.2k   	| 1 | 1 |
| [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/)     	| question-type classification 	| 6k      	| 0.5k    	| 1 | 1 |
| [SICK-E](http://clic.cimec.unitn.it/composes/sick.html)   	| natural language inference 	| 4.5k    	| 4.9k   	| 1 | 1 |
| [SNLI](https://nlp.stanford.edu/projects/snli/)     	| natural language inference   	| 550k    	| 9.8k   	| 1 | 1 |
| [MRPC](https://aclweb.org/aclwiki/Paraphrase_Identification_(State_of_the_art)) | paraphrase detection  | 4.1k | 1.7k | 1 | 1 |
| [STS 2012](https://www.cs.york.ac.uk/semeval-2012/task6/) 	| semantic textual similarity  	| N/A     	| 3.1k   	| 0  | 0 |
| [STS 2013](http://ixa2.si.ehu.es/sts/) 	| semantic textual similarity  	| N/A     	| 1.5k   	| 0  | 0 |
| [STS 2014](http://alt.qcri.org/semeval2014/task10/) 	| semantic textual similarity  	| N/A     	| 3.7k   	| 0  | 0 |
| [STS 2015](http://alt.qcri.org/semeval2015/task2/) 	| semantic textual similarity  	| N/A     	| 8.5k   	| 0  | 0 |
| [STS 2016](http://alt.qcri.org/semeval2016/task1/) 	| semantic textual similarity  	| N/A     	| 9.2k   	| 0  | 0 |
| [STS B](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark#Results)    	| semantic textual similarity  	| 5.7k    	| 1.4k   	| 1 | 0 |
| [SICK-R](http://clic.cimec.unitn.it/composes/sick.html)   	| semantic textual similarity | 4.5k    	| 4.9k   	| 1 | 0 |
| [COCO](http://mscoco.org/)     	| image-caption retrieval      	| 567k    	| 5*1k   	| 1 | 0 |

where **needs_train** means a model with parameters is learned on top of the sentence embeddings, and **set_classifier** means you can define the parameters of the classifier in the case of a classification task (see below).

Note: COCO comes with ResNet-101 2048d image embeddings. [More details on the tasks.](https://arxiv.org/pdf/1705.02364.pdf)

### Probing tasks
SentEval also includes a series of [*probing* tasks](https://github.com/facebookresearch/SentEval/tree/master/data/probing) to evaluate what linguistic properties are encoded in your sentence embeddings:

| Task     	| Type                         	| #train 	| #test 	| needs_train 	| set_classifier |
|----------	|------------------------------	|-----------:|----------:|:-----------:|:----------:|
| [SentLen](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Length prediction	| 100k     	| 10k    	| 1 | 1 |
| [WC](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Word Content analysis	| 100k     	| 10k    	| 1 | 1 |
| [TreeDepth](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Tree depth prediction	| 100k     	| 10k    	| 1 | 1 |
| [TopConst](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Top Constituents prediction	| 100k     	| 10k    	| 1 | 1 |
| [BShift](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Word order analysis	| 100k     	| 10k    	| 1 | 1 |
| [Tense](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Verb tense prediction	| 100k     	| 10k    	| 1 | 1 |
| [SubjNum](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Subject number prediction	| 100k     	| 10k    	| 1 | 1 |
| [ObjNum](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Object number prediction	| 100k     	| 10k    	| 1 | 1 |
| [SOMO](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Semantic odd man out	| 100k     	| 10k    	| 1 | 1 |
| [CoordInv](https://github.com/facebookresearch/SentEval/tree/master/data/probing)	| Coordination Inversion | 100k     	| 10k    	| 1 | 1 |

## Download datasets
To get all the transfer tasks datasets, run (in data/downstream/):
```bash
./get_transfer_data.bash
```
This will automatically download and preprocess the downstream datasets, and store them in data/downstream (warning: for MacOS users, you may have to use p7zip instead of unzip). The probing tasks are already in data/probing by default.

## How to use SentEval: examples

### examples/bow.py

In examples/bow.py, we evaluate the quality of the average of word embeddings.

To download state-of-the-art fastText embeddings:

```bash
curl -Lo glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
curl -Lo crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
```

To reproduce the results for bag-of-vectors, run (in examples/):  
```bash
python bow.py
```

As required by SentEval, this script implements two functions: **prepare** (optional) and **batcher** (required) that turn text sentences into sentence embeddings. Then SentEval takes care of the evaluation on the transfer tasks using the embeddings as features.

### examples/infersent.py

To get the **[InferSent](https://www.github.com/facebookresearch/InferSent)** model and reproduce our results, download our best models and run infersent.py (in examples/):
```bash
curl -Lo examples/infersent1.pkl https://dl.fbaipublicfiles.com/senteval/infersent/infersent1.pkl
curl -Lo examples/infersent2.pkl https://dl.fbaipublicfiles.com/senteval/infersent/infersent2.pkl
```

### examples/skipthought.py - examples/gensen.py - examples/googleuse.py

We also provide example scripts for three other encoders:

* [SkipThought with Layer-Normalization](https://github.com/ryankiros/layer-norm#skip-thoughts) in Theano
* [GenSen encoder](https://github.com/Maluuba/gensen) in Pytorch
* [Google encoder](https://tfhub.dev/google/universal-sentence-encoder/1) in TensorFlow

Note that for SkipThought and GenSen, following the steps of the associated githubs is necessary.
The Google encoder script should work as-is.

## How to use SentEval

To evaluate your sentence embeddings, SentEval requires that you implement two functions:

1. **prepare** (sees the whole dataset of each task and can thus construct the word vocabulary, the dictionary of word vectors etc)
2. **batcher** (transforms a batch of text sentences into sentence embeddings)


### 1.) prepare(params, samples) (optional)

*batcher* only sees one batch at a time while the *samples* argument of *prepare* contains all the sentences of a task.

```
prepare(params, samples)
```
* *params*: senteval parameters.
* *samples*: list of all sentences from the tranfer task.
* *output*: No output. Arguments stored in "params" can further be used by *batcher*.

*Example*: in bow.py, prepare is is used to build the vocabulary of words and construct the "params.word_vect* dictionary of word vectors.


### 2.) batcher(params, batch)
```
batcher(params, batch)
```
* *params*: senteval parameters.
* *batch*: numpy array of text sentences (of size params.batch_size)
* *output*: numpy array of sentence embeddings (of size params.batch_size)

*Example*: in bow.py, batcher is used to compute the mean of the word vectors for each sentence in the batch using params.word_vec. Use your own encoder in that function to encode sentences.

### 3.) evaluation on transfer tasks

After having implemented the batch and prepare function for your own sentence encoder,

1) to perform the actual evaluation, first import senteval and set its parameters:
```python
import senteval
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
```

2) (optional) set the parameters of the classifier (when applicable):
```python
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
```
You can choose **nhid=0** (Logistic Regression) or **nhid>0** (MLP) and define the parameters for training.

3) Create an instance of the class SE:
```python
se = senteval.engine.SE(params, batcher, prepare)
```

4) define the set of transfer tasks and run the evaluation:
```python
transfer_tasks = ['MR', 'SICKEntailment', 'STS14', 'STSBenchmark']
results = se.eval(transfer_tasks)
```
The current list of available tasks is:
```python
['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
```

## SentEval parameters
Global parameters of SentEval:
```bash
# senteval parameters
task_path                   # path to SentEval datasets (required)
seed                        # seed
usepytorch                  # use cuda-pytorch (else scikit-learn) where possible
kfold                       # k-fold validation for MR/CR/SUB/MPQA.
```

Parameters of the classifier:
```bash
nhid:                       # number of hidden units (0: Logistic Regression, >0: MLP); Default nonlinearity: Tanh
optim:                      # optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
tenacity:                   # how many times dev acc does not increase before training stops
epoch_size:                 # each epoch corresponds to epoch_size pass on the train set
max_epoch:                  # max number of epoches
dropout:                    # dropout for MLP
```

Note that to get a proxy of the results while **dramatically reducing computation time**,
we suggest the **prototyping config**:
```python
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
```
which will results in a 5 times speedup for classification tasks.

To produce results that are **comparable to the literature**, use the **default config**:
```python
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
```
which takes longer but will produce better and comparable results.

For probing tasks, we used an MLP with a Sigmoid nonlinearity and and tuned the nhid (in [50, 100, 200]) and dropout (in [0.0, 0.1, 0.2]) on the dev set.

## References

Please considering citing [[1]](https://arxiv.org/abs/1803.05449) if using this code for evaluating sentence embedding methods.

### SentEval: An Evaluation Toolkit for Universal Sentence Representations

[1] A. Conneau, D. Kiela, [*SentEval: An Evaluation Toolkit for Universal Sentence Representations*](https://arxiv.org/abs/1803.05449)

```
@article{conneau2018senteval,
  title={SentEval: An Evaluation Toolkit for Universal Sentence Representations},
  author={Conneau, Alexis and Kiela, Douwe},
  journal={arXiv preprint arXiv:1803.05449},
  year={2018}
}
```

Contact: [aconneau@fb.com](mailto:aconneau@fb.com), [dkiela@fb.com](mailto:dkiela@fb.com)

### Related work
* [J. R Kiros, Y. Zhu, R. Salakhutdinov, R. S. Zemel, A. Torralba, R. Urtasun, S. Fidler - SkipThought Vectors, NIPS 2015](https://arxiv.org/abs/1506.06726)
* [S. Arora, Y. Liang, T. Ma - A Simple but Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2017](https://openreview.net/pdf?id=SyK00v5xx)
* [Y. Adi, E. Kermany, Y. Belinkov, O. Lavi, Y. Goldberg - Fine-grained analysis of sentence embeddings using auxiliary prediction tasks, ICLR 2017](https://arxiv.org/abs/1608.04207)
* [A. Conneau, D. Kiela, L. Barrault, H. Schwenk, A. Bordes - Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, EMNLP 2017](https://arxiv.org/abs/1705.02364)
* [S. Subramanian, A. Trischler, Y. Bengio, C. J Pal - Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning, ICLR 2018](https://arxiv.org/abs/1804.00079)
* [A. Nie, E. D. Bennett, N. D. Goodman - DisSent: Sentence Representation Learning from Explicit Discourse Relations, 2018](https://arxiv.org/abs/1710.04334)
* [D. Cer, Y. Yang, S. Kong, N. Hua, N. Limtiaco, R. St. John, N. Constant, M. Guajardo-Cespedes, S. Yuan, C. Tar, Y. Sung, B. Strope, R. Kurzweil - Universal Sentence Encoder, 2018](https://arxiv.org/abs/1803.11175)
* [A. Conneau, G. Kruszewski, G. Lample, L. Barrault, M. Baroni - What you can cram into a single vector: Probing sentence embeddings for linguistic properties, ACL 2018](https://arxiv.org/abs/1805.01070)
