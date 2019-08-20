# Towards a Deep and Unified Understanding of Deep Neural Models in NLP

This submodule contains a tool for explaining hidden states of models. It is an implementation of the paper [*Towards a Deep and Unified Understanding of Deep Neural Models in NLP*](http://proceedings.mlr.press/v97/guan19a/guan19a.pdf)


## How to use

We provide a notebook tutorial [here](../../examples/interpret_NLP_models/understand_models.ipynb) to help you get started quickly. The main class needed is the `Interpreter` in [Interpreter.py](Interpreter.py). Given any input word embeddings and a forward function $\Phi$ that transforms the word embeddings $\bf x$ to a hidden state $\bf s$, the Interpreter helps understand how much each input word contributes to the hidden state. Suppose the $\Phi$, the input $\bf x$ and the input words are defined as:
```
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(5,256) / 100
x = x.to(device)
words = ['1','2','3','4','5']

def Phi(x):
    W = torch.tensor([10., 20., 5., -20., -10.]).to(device)
    return W @ x
```

To explain a certain hidden state, we also need to get its variance for regularization. We provide a simple tool in `Interpreter.py` for calculating regularization. You just need to provide your sampled x as a list and your Phi. as shown below:

```
from Interpreter import calculate_regularization

# here we sample input x using random for simplicity
sampled_x = [torch.randn(5,256) / 100 for _ in range(100)]

regularization = calculate_regularization(sampled_x, Phi, device=device)
```

To explain this case, we need to initialize an `Interpreter` class, and pass $\bf x$, regularization and $\Phi$ to it (we also need to set hyper-parameter scale to a reasonable value: 10 * Std[embedding] is recommanded):
```
from Interpreter import Interpreter

interpreter = Interpreter(x=x, Phi=Phi, regularization=regularization, scale=10 * 0.1, words=words).to(device)
```
Then, we need the interpreter to optimize itself by minimizing the loss function in paper.
```
interpreter.optimize(iteration=5000, lr=0.5, show_progress=True)
```
After optimizing, we can get the best sigma:
```
interpreter.get_sigma()
```
the result will be something like:
```
array([0.00315634, 0.00181308, 0.00633237, 0.00174878, 0.0030807 ], dtype=float32)
```
Every sigma stands for the change limit of input without changing hidden state too much. The smaller the sigma is, the more this input word contributes to the hidden state.

Now, we can get the explanation by calling the visualize function:
```
interpreter.visualize()
```
Then, we can get results below:

![](https://nlpbp.blob.core.windows.net/images/result.png)

which means that the second and forth words are most important to $\Phi$, which is reasonable because the weight of them are larger.

## Explain a certain layer in any saved pytorch model

We provide an example on how to use our method to explain a saved pytorch model (*pre-trained BERT model in our case*) [here](../../examples/interpret_NLP_models/understand_models.ipynb). 
> NOTE: This result may not be consistent with the result in the paper because we use the pre-trained BERT model directly for simplicity, while the BERT model we use in paper is fine-tuned on a specific dataset like SST-2.
