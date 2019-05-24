# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This file is used by pytest to inject fixtures automatically. As it is explained in the documentation
# https://docs.pytest.org/en/latest/fixture.html:
# "If during implementing your tests you realize that you want to use a fixture function from multiple test files
# you can move it to a conftest.py file. You don’t need to import the fixture you want to use in a test, it
# automatically gets discovered by pytest."

import pytest
from tests.notebooks_common import path_notebooks


@pytest.fixture(scope="module")
def notebooks():
    folder_notebooks = path_notebooks()

    # Path for the notebooks
    paths = {
        "msrpc": os.path.join(
            folder_notebooks,"sentence_similarity", "01-prep-data", "msrpc.ipynb"
        ),
        "snli": os.path.join(
            folder_notebooks,"sentence_similarity", "01-prep-data", "snli.ipynb"
        ),
        "stsbenchmark": os.path.join(
            folder_notebooks,"sentence_similarity", "01-prep-data", "stsbenchmark.ipynb"
        ),

    }
    return paths