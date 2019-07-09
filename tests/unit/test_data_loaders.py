# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random

import numpy as np
import pytest
import json
import os
import io

from utils_nlp.dataset.data_loaders import DaskCSVLoader
from utils_nlp.dataset.data_loaders import DaskJSONLoader

UNIF1 = {"a": 4, "b": 6, "n": 10000}  # some uniform distribution
row_size = 5  # "a,b\n (5 bytes)"
json_row_size = 18  # "{"a": 1, "b": 5}\n (18 bytes)"


@pytest.fixture()
def csv_file(tmpdir):
    random.seed(0)
    f = tmpdir.mkdir("test_loaders").join("tl_data.csv")
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


@pytest.fixture()
def json_file(tmpdir):
    random.seed(0)
    json_path = os.path.join(tmpdir, "test.jsonl")
    with io.open(json_path, "w", encoding="utf8") as f:
        for _ in range(UNIF1["n"]):
            data_dict = {
                "a": random.randint(0, 1),
                "b": random.randint(UNIF1["a"], UNIF1["b"]),
            }
            json.dump(data_dict, f)
            f.write("\n")
    return json_path


def test_dask_csv_rnd_loader(csv_file):
    num_batches = 500
    batch_size = 12
    num_partitions = 4

    loader = DaskCSVLoader(
        csv_file,
        header=None,
        block_size=row_size * int(UNIF1["n"] / num_partitions),
        random_seed=0,
    )

    sample = []
    for batch in loader.get_random_batches(num_batches, batch_size):
        sample.append(list(batch.iloc[:, 1]))
    sample = np.concatenate(sample)

    assert loader.df.npartitions == num_partitions
    assert sample.mean().round() == (UNIF1["a"] + UNIF1["b"]) / 2
    assert len(sample) <= num_batches * batch_size


def test_dask_csv_seq_loader(csv_file):
    batch_size = 12
    num_partitions = 4

    loader = DaskCSVLoader(
        csv_file,
        header=None,
        block_size=row_size * int(UNIF1["n"] / num_partitions),
    )

    sample = []
    for batch in loader.get_sequential_batches(batch_size):
        sample.append(list(batch.iloc[:, 1]))
    sample = np.concatenate(sample)

    assert loader.df.npartitions == num_partitions
    assert sample.mean().round() == (UNIF1["a"] + UNIF1["b"]) / 2
    assert len(sample) == UNIF1["n"]


def test_dask_json_rnd_loader(json_file):
    num_batches = 500
    batch_size = 12
    num_partitions = 4

    loader = DaskJSONLoader(
        json_file,
        block_size=json_row_size * int(UNIF1["n"] / num_partitions),
        random_seed=0,
        lines=True,
    )

    sample = []
    for batch in loader.get_random_batches(num_batches, batch_size):
        sample.append(list(batch.iloc[:, 1]))
    sample = np.concatenate(sample)

    assert loader.df.npartitions == num_partitions
    assert sample.mean().round() == (UNIF1["a"] + UNIF1["b"]) / 2
    assert len(sample) <= num_batches * batch_size


def test_dask_json_seq_loader(json_file):
    batch_size = 12
    num_partitions = 4

    loader = DaskJSONLoader(
        json_file,
        block_size=json_row_size * int(UNIF1["n"] / num_partitions),
        random_seed=0,
        lines=True,
    )

    sample = []
    for batch in loader.get_sequential_batches(batch_size):
        sample.append(list(batch.iloc[:, 1]))
    sample = np.concatenate(sample)

    assert loader.df.npartitions == num_partitions
    assert sample.mean().round() == (UNIF1["a"] + UNIF1["b"]) / 2
    assert len(sample) == UNIF1["n"]
