import random

import pytest

import torch
from torch import nn

from utils_nlp.interpreter.Interpreter import (
    Interpreter,
    calculate_regularization,
)

lstm_layer = nn.LSTM(10, 10)


def fixed_length_Phi(x):
    return x[0] * 10 + x[1] * 20 - x[2] * 20 - x[3] * 10


def variable_length_Phi(x):
    return lstm_layer(x.unsqueeze(0))[0][0]


@pytest.fixture
def fixed_length_interp():
    x = torch.randn(4, 10)
    regular = torch.randn(10)
    return Interpreter(x, fixed_length_Phi, regularization=regular)


@pytest.fixture
def variable_length_interp():
    x = torch.randn(4, 10)
    regular = torch.randn(1, 10)
    return Interpreter(x, variable_length_Phi, regularization=regular)


def test_fixed_length_regularization():
    dataset = torch.randn(10, 4, 10)
    regular = calculate_regularization(dataset, fixed_length_Phi)
    assert regular.shape == (10,)


def test_variable_length_regularization():
    dataset = [torch.randn(random.randint(5, 9), 10) for _ in range(10)]
    regular = calculate_regularization(
        dataset, variable_length_Phi, reduced_axes=[0]
    )
    assert regular.shape == (1, 10)


def test_initialize_interpreter():
    x = torch.randn(4, 10)
    regular = torch.randn(10)
    _ = Interpreter(x, fixed_length_Phi, regularization=regular)


def test_train_fixed_length_interp(fixed_length_interp):
    fixed_length_interp.optimize(iteration=10)


def test_train_variable_length_interp(variable_length_interp):
    variable_length_interp.optimize(iteration=10)


def test_interpreter_get_simga(fixed_length_interp):
    sigma = fixed_length_interp.get_sigma()
    assert sigma.shape == (4,)
