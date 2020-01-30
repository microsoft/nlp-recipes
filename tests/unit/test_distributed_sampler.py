# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#import sys
#sys.path.insert(0, "../../")
from utils_nlp.models.transformers.extractive_summarization import IterableDistributedSampler

@pytest.mark.cpu
def test_sampler():
    sampler = IterableDistributedSampler(1, -1)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'abcdefg'

    sampler = IterableDistributedSampler(2, -1)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'abcdefg'

    sampler = IterableDistributedSampler(4, 1)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'bf'

    sampler = IterableDistributedSampler(4, 2)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'cg'

    sampler = IterableDistributedSampler(4, 3)
    samples = list(sampler.iter('abcdefg'))
    assert ''.join(samples) == 'd'

