# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='SentEval',
    version='0.1.0',
    url='https://github.com/facebookresearch/SentEval',
    packages=find_packages(exclude=['examples']),
    license='Attribution-NonCommercial 4.0 International',
    long_description=readme,
)
