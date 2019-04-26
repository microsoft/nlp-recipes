# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from gensim.models.keyedvectors import KeyedVectors


def load_word2vec():
    # Todo : Move to azure blob and get rid of this path.
    file_path = (
        "../../../Pretrained Vectors/GoogleNews-vectors-negative300.bin"
    )
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    print(type(model))


if __name__ == "__main__":
    load_word2vec()
