# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from setuptools_scm import get_version


__title__ = "Microsoft NLP"
__author__ = "AI CAT at Microsoft"
__license__ = "MIT"
__copyright__ = "Copyright 2018-present Microsoft Corporation"

# Synonyms
TITLE = __title__
AUTHOR = __author__
LICENSE = __license__
COPYRIGHT = __copyright__

# Determine semantic versioning automatically
# from git commits if the package is installed
# into your environment, otherwise
# we set version to default for development
try:
    __version__ = get_version()
except LookupError:
    __version__ = "0.0.0"

VERSION = __version__

