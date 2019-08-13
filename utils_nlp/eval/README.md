### [Evaluation (Eval)](eval)
The evaluation (eval) submodule includes functionality for computing common classification and evaluation metrics like accuracy, precision, recall, and f1 scores for classification scenarios, normalizing and finding f1_scores for different datasets like SQuAD, as well as logging the means and other coefficients for datasets like senteval.  

It contains the following scripts and wrappers:  

### Scripts  
1. [classification.py](./classification.py) - This defines common classification evaluation metrics, and computes correlation coefficients (Pearson product-moment correlation).   

1. [evaluate_squad.py](./evaluate_squad.py) - This is the official evaluation script for the SQuAD v1.1 dataset. It exposes functionalities to normalize your answers, return f1 scores, find exact match, and determine maximum over ground truth metrics. Original source can be found [here](https://github.com/allenai/bi-att-flow/blob/498c8026d92a8bcf0286e2d216d092d444d02d76/squad/evaluate-v1.1.py)  

1. [senteval.py](./senteval.py) - This defines an object `SentEvalConfig` which is a store for all properties generated from sentence evaluation experiements.  
This script

1. [Sentence Evaluation Toolkit](./SentEval/README.md) - SentEval is a library for evaluating the quality of sentence embeddings. We assess their generalization power by using them as features on a broad and diverse set of "transfer" tasks. 