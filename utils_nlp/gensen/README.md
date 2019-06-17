# GenSen

Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning

Sandeep Subramanian, Adam Trischler, Yoshua Bengio & Christopher Pal

ICLR 2018


### About

GenSen is a technique to learn general purpose, fixed-length representations of sentences via multi-task training. These representations are useful for transfer and low-resource learning. For details please refer to ICLR [paper](https://openreview.net/forum?id=B18WgG-CZ&noteId=B18WgG-CZ).

### Code

We provide a distributed PyTorch with Horovod implementation of the paper along with pre-trained models as well as code to evaluate these models on a variety of transfer learning benchmarks.
This code is based on the gibhub codebase from [Maluuba](https://github.com/Maluuba/gensen), but we have refactored the code in the following aspects:
1. Support a distributed PyTorch with Horovod
2. Clean and refactor the original code in a more structured form
3. Change the training file (`train.py`) from non-stopping to stop when the validation loss reaches to the local minimum
4. Update the code from Python 2.7 to 3+ and PyTorch from 0.2 or 0.3 to 1.0.1
5. Add some necessary comments
6. Add some code for training on AzureML platform
7. Fix the bug on when setting the batch size to 1, the training raises an error
### Requirements

- Python 3+
- PyTorch 1.0.1
- nltk
- h5py
- numpy
- scikit-learn

### Reference

```
@article{subramanian2018learning,
title={Learning general purpose distributed sentence representations via large scale multi-task learning},
author={Subramanian, Sandeep and Trischler, Adam and Bengio, Yoshua and Pal, Christopher J},
journal={arXiv preprint arXiv:1804.00079},
year={2018}
}
```
