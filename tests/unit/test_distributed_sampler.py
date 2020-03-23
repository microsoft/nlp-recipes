# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from utils_nlp.models.transformers.extractive_summarization import IterableDistributedSampler

@pytest.mark.cpu
def test_sampler():
    sampler = IterableDistributedSampler(1, 0, -1)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'abcdefg'

    sampler = IterableDistributedSampler(2, 0, -1)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'abcdefg'

    sampler = IterableDistributedSampler(4, 1, 1)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'bf'

    sampler = IterableDistributedSampler(4, 2, 2)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'cg'

    sampler = IterableDistributedSampler(4, 3, 3)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'd'

    sampler = IterableDistributedSampler(8, 7, 3)
    samples = list(sampler.iter('abcdefghijklmn'))
    assert ''.join(samples) == 'h'


