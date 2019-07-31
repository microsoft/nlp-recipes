#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io

import re
from glob import glob
from os.path import basename, dirname, join, splitext

from setuptools import find_packages, setup

VERSION = __import__("__init__").VERSION


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fh:
        return fh.read()


setup(
    name="utils_nlp",
    version=VERSION,
    license="MIT License",
    description="NLP Utility functions that are used for best practices in building state-of-the-art NLP methods and scenarios. Developed by Microsoft AI CAT",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.md")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CONTRIBUTING.md")),
    ),
    author="AI CAT",
    author_email="teamsharat@microsoft.com",
    url="https://github.com/microsoft/nlp",
    packages=["utils_nlp"],
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Telecommunications Industry",
    ],
    project_urls={
        "Documentation": "https://github.com/microsoft/nlp/",
        "Issue Tracker": "https://github.com/microsoft/nlp/issues",
    },
    keywords=[
        "Microsoft NLP",
        "Natural Language Processing",
        "Text Processing",
        "Word Embedding",
    ],
    python_requires=">=3.6",
    install_requires=[],
    dependency_links=[],
    extras_require={},
    setup_requires=[],
)
