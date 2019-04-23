# GenSen

Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning

Sandeep Subramanian, Adam Trischler, Yoshua Bengio & Christopher Pal

ICLR 2018


### About

GenSen is a technique to learn general purpose, fixed-length representations of sentences via multi-task training. These representations are useful for transfer and low-resource learning. For details please refer to our ICLR [paper](https://openreview.net/forum?id=B18WgG-CZ&noteId=B18WgG-CZ).

### Code

We provide a PyTorch implementation of our paper along with pre-trained models as well as code to evaluate these models on a variety of transfer learning benchmarks.

### Requirements

- Python 2.7 (Python 3 compatibility coming soon)
- PyTorch 0.2 or 0.3
- nltk
- h5py
- numpy
- scikit-learn

#### Usage

##### Setting up Models & pre-trained word vecotrs

You download our pre-trained models and set up pre-trained word vectors for vocabulary expansion by

```bash
cd data/models
bash download_models.sh
cd ../embedding
bash glove2h5.sh
```

##### Using a pre-trained model to extract sentence representations.

You can use our pre-trained models to extract the last hidden state or all hidden states of our multi-task GRU. Additionally, you can concatenate the output of multiple models to replicate the numbers in our paper.

```python
from gensen import GenSen, GenSenSingle

gensen_1 = GenSenSingle(
    model_folder='./data/models',
    filename_prefix='nli_large_bothskip',
    pretrained_emb='./data/embedding/glove.840B.300d.h5'
)
reps_h, reps_h_t = gensen_1.get_representation(
    sentences, pool='last', return_numpy=True, tokenize=True
)
print reps_h.shape, reps_h_t.shape
```

- The input to `get_representation` is `sentences`, which should be a list of strings. If your strings are not pre-tokenized, then set `tokenize=True` to use the NLTK tokenizer before computing representations.
- `reps_h` (batch_size x seq_len x 2048) contains the hidden states for all words in all sentences (padded to the max length of sentences)
- `reps_h_t` (batch_size x 2048) contains only the last hidden state for all sentences in the minibatch 

GenSenSingle will return the output of a single model `nli_large_bothskip (+STN +Fr +De +NLI +L +STP)`. You can concatenate the output of multiple models by creating a GenSen instance with multiple GenSenSingle instances, as follows:

```python
gensen_2 = GenSenSingle(
    model_folder='./data/models',
    filename_prefix='nli_large_bothskip_parse',
    pretrained_emb='./data/embedding/glove.840B.300d.h5'
)
gensen = GenSen(gensen_1, gensen_2)
reps_h, reps_h_t = gensen.get_representation(
    sentences, pool='last', return_numpy=True, tokenize=True
)
```

1) `reps_h` (batch_size x seq_len x 4096) contains the hidden states for all words in all sentences (padded to the max length of sentences)
2) `reps_h_t` (batch_size x 4096) contains only the last hidden state for all sentences in the minibatch 

The model will produce a fixed-length vector for each sentence as well as the hidden states corresponding to each word in every sentence (padded to max sentence length). You can also return a numpy array instead of a `torch.FloatTensor` by setting `return_numpy=True`. 

##### Vocabulary Expansion

If you have a specific domain for which you want to compute representations, you can call `vocab_expansion` on instances of the GenSenSingle or GenSen class simply by `gensen.vocab_expansion(vocab)` where vocab is a list of unique words in the new domain. This will learn a linear mapping from the provided pretrained embeddings (which have a significantly larger vocabulary) provided to the space of gensen's word vectors. For an example of how this is used in an actual setting, please refer to `gensen_senteval.py`.

##### Training a model from scratch

To train a model from scratch, simply run `train.py` with an appropriate JSON config file. An example config is provided in `example_config.json`. To continue training, just relaunch the same scripy with `load_dir=auto` in the config file.

To download some of the data required to train a GenSen model, run:

```bash
bash get_data.sh
```

Note that this script can take a while to complete since it downloads, tokenizes and lowercases a fairly large En-Fr corpus. If you already have these parallel corpora processed, you can replace the paths to these files in the provided `example_config.json` 

Some of the data used in our work is no longer publicly available (BookCorpus - see http://yknzhu.wixsite.com/mbweb) or has an LDC license associated (Penn Treebank). As a result, the `example_config.json` script will only train on Multilingual NMT and NLI, since they are publicly available. To use models trained on all tasks, please use our available pre-trained models.

Additional Sequence-to-Sequence transduction tasks can be added trivally to the multi-task framework by editing the json config file with more tasks.

```bash
python train.py --config example_config.json
```

To use the default settings in `example_config.json` you will need a GPU with atleast 16GB of memory (such as a P100), to train on smaller GPUs, you may need to reduce the batch size.

Note that if "load_dir" is set to auto, the script will resume from the last saved model in "save_dir".

##### Creating a GenSen model from a trained multi-task model

Once you have a trained model, we can throw away all of the decoders and just retain the encoder used to compute sentence representations.

You can do this by running

```bash
python create_gensen.py -t <path_to_trained_model> -s <path_to_save_encoder> -n <name_of_encoder>
```

Once you have done this, you can load this model just like any of the pre-trained models by specifying the model_folder as `path_to_save_encoder` and filename_prefix as `name_of_encoder` in the above command.

```python
your_gensen = GenSenSingle(
    model_folder='<path_to_save_encoder>',
    filename_prefix='<name_of_encoder>',
    pretrained_emb='./data/embedding/glove.840B.300d.h5'
)
```

### Transfer Learning Evaluations

We used the [SentEval](https://github.com/facebookresearch/SentEval) toolkit to run most of our transfer learning experiments. To replicate these numbers, clone their repository and follow setup instructions. Once complete, copy `gensen_senteval.py` and `gensen.py` into their examples folder and run the following commands to reproduce different rows in Table 2 of our paper. Note: Please set the path to the pretrained glove embeddings (`glove.840B.300d.h5`) and model folder as appropriate.

```
(+STN +Fr +De +NLI +L +STP)      python gensen_senteval.py --prefix_1 nli_large --prefix_2 nli_large_bothskip
(+STN +Fr +De +NLI +2L +STP)     python gensen_senteval.py --prefix_1 nli_large_bothskip --prefix_2 nli_large_bothskip_2layer
(+STN +Fr +De +NLI +L +STP +Par) python gensen_senteval.py --prefix_1 nli_large_bothskip_parse --prefix_2 nli_large_bothskip
```

### Reference

```
@article{subramanian2018learning,
title={Learning general purpose distributed sentence representations via large scale multi-task learning},
author={Subramanian, Sandeep and Trischler, Adam and Bengio, Yoshua and Pal, Christopher J},
journal={arXiv preprint arXiv:1804.00079},
year={2018}
}
```
