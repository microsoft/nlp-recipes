# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random

import numpy as np
import pytest

from utils_nlp.dataset.data_loaders import DaskCSVLoader

UNIF1 = {"a": 0, "b": 10, "n": 1000}


@pytest.fixture()
def csv_file(tmpdir):
    f = tmpdir.mkdir("test_loaders").join("tl_data.csv")
    for i in range(1000):
        f.write(
            "\n".join(
                [
                    "{},{}".format(
                        random.randint(0, 1),
                        random.randint(UNIF1["a"], UNIF1["b"]),
                    )
                    for x in range(UNIF1["n"])
                ]
            )
        )
    return str(f)


def test_dask_csv_loader(csv_file):
    num_batches = 500
    batch_size = 12
    num_partitions = 4

    loader = DaskCSVLoader(
        csv_file, header=None, block_size=5 * int(UNIF1["n"] / num_partitions)
    )

    sample = []
    for batch in loader.get_random_batches(num_batches, batch_size):
        sample.append(list(batch.iloc[:, 1]))
    sample = np.concatenate(sample)

    assert loader.df.npartitions == num_partitions
    assert sample.mean().round() == UNIF1["a"] + UNIF1["b"] / 2
    assert len(sample) <= num_batches * batch_size
