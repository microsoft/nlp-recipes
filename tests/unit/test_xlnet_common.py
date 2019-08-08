# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

def test_preprocess_classification_tokens(xlnet_english_tokenizer):
    text = ["Hello World.",
            "How you doing?",
            "greatttt",
            "The quick, brown fox jumps over a lazy dog.",
            " DJs flock by when MTV ax quiz prog",
            "Quick wafting zephyrs vex bold Jim",
            "Quick, Baz, get my woven flax jodhpurs!"            
           ]
    seq_length = 5
    input_ids, input_mask, segment_ids = xlnet_english_tokenizer.preprocess_classification_tokens(text, seq_length)
    
    assert len(input_ids) == len(text)
    assert len(input_mask) == len(text)
    assert len(segment_ids) == len(text)
    
    
    for sentence in range(len(text)):
        assert len(input_ids[sentence]) == seq_length
        assert len(input_mask[sentence]) == seq_length
        assert len(segment_ids[sentence]) == seq_length
    