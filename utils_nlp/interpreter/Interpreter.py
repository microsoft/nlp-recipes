# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utilities that enables you to explain every hidden state in your model"""

import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def calculate_regularization(sampled_x, Phi, reduced_axes=None, device=None):
    """ Calculate the variance that is used for Interpreter

    Args:
        sampled_x (list of torch.FloatTensor):
            A list of sampled input embeddings $x$, each $x$ is of shape
            ``[length, dimension]``. All the $x$s can have different length,
            but should have the same dimension. Sampled number should be
            higher to get a good estimation.
        reduced_axes (list of ints, Optional):
            The axes that is variable in Phi (e.g., the sentence length axis).
            We will reduce these axes by mean along them.

    Returns:
        torch.FloatTensor: The regularization term calculated

    """
    sample_num = len(sampled_x)
    sample_s = []
    for n in range(sample_num):
        x = sampled_x[n]
        if device is not None:
            x = x.to(device)
        s = Phi(x)
        if reduced_axes is not None:
            for axis in reduced_axes:
                assert axis < len(s.shape)
                s = s.mean(dim=axis, keepdim=True)
        sample_s.append(s.tolist())
    sample_s = np.array(sample_s)
    return np.std(sample_s, axis=0)


class Interpreter(nn.Module):
    """ Interpreter for interpreting one instance. The method is from
    paper `Towards a Deep and Unified Understanding of Deep Neural
    Models in NLP <http://proceedings.mlr.press/v97/guan19a/guan19a.pdf>`_

    It will minimize the loss in Eqn.(7):

        $L(sigma) = (||Phi(embed + epsilon) - Phi(embed)||_2^2)
        // (regularization^2) - rate * log(sigma)$

    In our implementation, we use reparameterization trick to represent
    epsilon ~ N(0, sigma^2 I), i.e. epsilon = scale * ratio * noise.
    Where noise ~ N(0, 1), scale is a hyper-parameter that controls the
    maximum value of sigma^2, and ratio in (0, 1) is the learnable parameter.

    """

    def __init__(self, x, Phi, scale=0.5, rate=0.1, regularization=None, words=None):
        """ Initialize an interpreter class.

        Args:
            x (torch.FloatTensor): Of shape ``[length, dimension]``.
                The $x$ we studied. i.e. The input word embeddings.
            Phi (function):
                The $Phi$ we studied. A function whose input is x (the first
                parameter) and returns a hidden state (of type
                ``torch.FloatTensor``, of any shape)
            scale (float):
                The maximum size of sigma. A hyper-parameter in
                reparameterization trick. The recommended value is
                10 * Std[word_embedding_weight], where word_embedding_weight
                is the word embedding weight in the model interpreted. Larger
                scale will give more salient result, Default: 0.5.
            rate (float):
                A hyper-parameter that balance the MLE Loss and Maximum
                Entropy Loss. Larger rate will result in larger information
                loss. Default: 0.1.
            regularization (Torch.FloatTensor or np.ndarray):
                The regularization term, should be of the same shape as
                (or broadcastable to) the output of Phi. If None is given,
                method will use the output to regularize itself.
                Default: None.
            words (List[Str]):
                The input sentence, used for visualizing. If None is given,
                method will not show the words.

        """
        super(Interpreter, self).__init__()
        self.s = x.size(0)
        self.d = x.size(1)
        self.ratio = nn.Parameter(torch.randn(self.s, 1), requires_grad=True)

        self.scale = scale
        self.rate = rate
        self.x = x
        self.Phi = Phi

        self.regular = regularization
        if self.regular is not None:
            self.regular = nn.Parameter(torch.tensor(self.regular).to(x), requires_grad=False)
        self.words = words
        if self.words is not None:
            assert self.s == len(
                words
            ), "the length of x should be of the same with the lengh of words"

    def forward(self):
        """ Calculate loss:

            $L(sigma) = (||Phi(embed + epsilon) - Phi(embed)||_2^2)
            // (regularization^2) - rate * log(sigma)$

        Returns:
            torch.FloatTensor: a scalar, the target loss.

        """
        ratios = torch.sigmoid(self.ratio)  # S * 1
        x = self.x + 0.0  # S * D
        x_tilde = x + ratios * torch.randn(self.s, self.d).to(x.device) * self.scale  # S * D
        s = self.Phi(x)  # D or S * D
        s_tilde = self.Phi(x_tilde)
        loss = (s_tilde - s) ** 2
        if self.regular is not None:
            loss = torch.mean(loss / self.regular ** 2)
        else:
            loss = torch.mean(loss) / torch.mean(s ** 2)

        return loss - torch.mean(torch.log(ratios)) * self.rate

    def optimize(self, iteration=5000, lr=0.01, show_progress=False):
        """ Optimize the loss function

        Args:
            iteration (int): Total optimizing iteration
            lr (float): Learning rate
            show_progress (bool): Whether to show the learn progress

        """
        minLoss = None
        state_dict = None
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        func = (lambda x: x) if not show_progress else tqdm
        for _ in func(range(iteration)):
            optimizer.zero_grad()
            loss = self()
            loss.backward()
            optimizer.step()
            if minLoss is None or minLoss > loss:
                state_dict = {k: self.state_dict()[k] + 0.0 for k in self.state_dict().keys()}
                minLoss = loss
        self.eval()
        self.load_state_dict(state_dict)

    def get_sigma(self):
        """ Calculate and return the sigma

        Returns:
            np.ndarray: of shape ``[seqLen]``, the ``sigma``.

        """
        ratios = torch.sigmoid(self.ratio)  # S * 1
        return ratios.detach().cpu().numpy()[:, 0] * self.scale

    def visualize(self):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_sigma()
        _, ax = plt.subplots()
        im = ax.imshow([sigma_], cmap="GnBu_r")
        ax.set_xticks(range(self.s))
        ax.set_xticklabels(self.words)
        ax.set_yticks([0])
        ax.set_yticklabels([""])
        plt.colorbar(im, orientation="horizontal")
        plt.tight_layout()
        plt.show()
