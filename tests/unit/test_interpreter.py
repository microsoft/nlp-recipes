# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import random

import pytest

import numpy as np
import torch
from torch import nn

from utils_nlp.interpreter.Interpreter import (
    Interpreter,
    calculate_regularization,
)


def fixed_length_Phi(x):
    return x[0] * 10 + x[1] * 20 - x[2] * 20 - x[3] * 10


def variable_length_Phi(function):
    return lambda x: (function(x.unsqueeze(0))[0][0])


@pytest.fixture
def fixed_length_interp():
    x = torch.randn(4, 10)
    regular = torch.randn(10)
    return Interpreter(x, fixed_length_Phi, regularization=regular)


@pytest.fixture
def variable_length_interp():
    function = nn.LSTM(10, 10)
    x = torch.randn(4, 10)
    regular = torch.randn(1, 10)
    return Interpreter(
        x, variable_length_Phi(function), regularization=regular
    )


def test_fixed_length_regularization():
    dataset = torch.randn(10, 4, 10)
    # calculate all hidden states
    hidden = [fixed_length_Phi(x).tolist() for x in dataset]
    # calculate the standard deviation
    hidden = np.array(hidden)
    regular_gt = np.std(hidden, axis=0)
    regular = calculate_regularization(dataset, fixed_length_Phi)
    assert np.sum(np.abs(regular - regular_gt)) < 1e-5


def test_variable_length_regularization():
    function = nn.LSTM(10, 10)
    dataset = [torch.randn(random.randint(5, 9), 10) for _ in range(10)]
    # calculate all hidden states
    hidden = [
        np.mean(
            variable_length_Phi(function)(x).tolist(), axis=0, keepdims=True
        )
        for x in dataset
    ]
    # calculate the standard deviation
    hidden = np.array(hidden)
    regular_gt = np.std(hidden, axis=0)
    regular = calculate_regularization(
        dataset, variable_length_Phi(function), reduced_axes=[0]
    )
    assert np.sum(np.abs(regular - regular_gt)) < 1e-5


def test_initialize_interpreter():
    x = torch.randn(4, 10)
    regular = torch.randn(10)
    interpreter = Interpreter(x, fixed_length_Phi, regularization=regular)
    assert interpreter.s == 4
    assert interpreter.d == 10
    assert interpreter.regular.tolist() == regular.tolist()


def test_train_fixed_length_interp(fixed_length_interp):
    init_ratio = fixed_length_interp.ratio + 0.0  # make a copy
    init_regular = fixed_length_interp.regular + 0.0
    fixed_length_interp.optimize(iteration=10)
    after_ratio = fixed_length_interp.ratio + 0.0
    after_regular = fixed_length_interp.regular + 0.0
    # make sure the ratio is changed when optimizing
    assert torch.sum(torch.abs(after_ratio - init_ratio)) > 1e-5
    # make sure the regular is not changed when optimizing
    assert torch.sum(torch.abs(after_regular - init_regular)) < 1e-5


def test_train_variable_length_interp(variable_length_interp):
    init_ratio = variable_length_interp.ratio + 0.0  # make a copy
    init_regular = variable_length_interp.regular + 0.0
    variable_length_interp.optimize(iteration=10)
    after_ratio = variable_length_interp.ratio + 0.0
    after_regular = variable_length_interp.regular + 0.0
    # make sure the ratio is changed when optimizing
    assert torch.sum(torch.abs(after_ratio - init_ratio)) > 1e-5
    # make sure the regular is not changed when optimizing
    assert torch.sum(torch.abs(after_regular - init_regular)) < 1e-5


def test_interpreter_get_simga(fixed_length_interp):
    sigma = fixed_length_interp.get_sigma()
    assert sigma.shape == (4,)
