# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading, extracting, and reading the
    CNN/DM dataset at https://github.com/harvardnlp/sent-summary.
"""

import glob
import itertools
import nltk

nltk.download("punkt")
from nltk import tokenize
import os
from os.path import isfile, join
import sys
import regex as re
import torch
import torchtext
from torchtext.utils import download_from_url, extract_archive
import zipfile

from bertsum.others.utils import clean

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.models.transformers.extractive_summarization import get_dataset, get_dataloader


def _line_iter(file_path):
    with open(file_path, "r", encoding="utf8") as fd:
        for line in fd:
            yield line


def _create_data_from_iterator(iterator, preprocessing, word_tokenizer):
    # data = []
    # for line in iterator:
    #    data.append(preprocess((line, preprocessing, word_tokenizer)))
    # return data
    for line in iterator:
        yield preprocess((line, preprocessing, word_tokenizer))


def _remove_ttags(line):
    line = re.sub(r"<t>", "", line)
    # change </t> to <q>
    # pyrouge test requires <q> as sentence splitter
    line = re.sub(r"</t>", "<q>", line)
    return line


def _cnndm_target_sentence_tokenization(line):
    return line.split("<q>")


def preprocess(param):
    """
    Helper function to preprocess a list of paragraphs.

    Args:
        param (Tuple): params are tuple of (a list of strings, a list of preprocessing functions, and function to tokenize setences into words). A paragraph is represented with a single string with multiple setnences.

    Returns:
        list of list of strings, where each string is a token or word.
    """

    sentences, preprocess_pipeline, word_tokenize = param
    for function in preprocess_pipeline:
        sentences = function(sentences)
    return [word_tokenize(sentence) for sentence in sentences]


class Summarization(torch.utils.data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.source)

    def __init__(
        self,
        source_file,
        target_file,
        source_preprocessing,
        target_preprocessing,
        word_tokenization,
        top_n=-1,
        **kwargs,
    ):
        """ create an CNN/CM dataset instance given the paths of source file and target file"""

        super(Summarization, self).__init__()
        source_iter = _line_iter(source_file)
        target_iter = _line_iter(target_file)

        if top_n != -1:
            source_iter = itertools.islice(source_iter, top_n)
            target_iter = itertools.islice(target_iter, top_n)

        self._source = _create_data_from_iterator(
            source_iter, source_preprocessing, word_tokenization
        )

        self._target = _create_data_from_iterator(
            target_iter, target_preprocessing, word_tokenization
        )

    # def __getitem__(self, i):
    #    return self._source[i]

    # def __len__(self):
    #    return len(self._source)

    def __iter__(self):
        for x in self._source:
            yield x

    def get_target(self):
        return self._target


def CNNDMSummarization(*args, **kwargs):
    urls = ["https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz"]
    dirname = "cnndmsum"
    name = "cnndmsum"

    def _setup_datasets(url, top_n=-1, local_cache_path=".data"):
        file_name = "cnndm.tar.gz"
        maybe_download(url, file_name, local_cache_path)
        dataset_tar = os.path.join(local_cache_path, file_name)
        extracted_files = extract_archive(dataset_tar)
        for fname in extracted_files:
            if fname.endswith("train.txt.src"):
                train_source_file = fname
            if fname.endswith("train.txt.tgt.tagged"):
                train_target_file = fname
            if fname.endswith("test.txt.src"):
                test_source_file = fname
            if fname.endswith("test.txt.tgt.tagged"):
                test_target_file = fname

        return (
            Summarization(
                train_source_file,
                train_target_file,
                [clean, tokenize.sent_tokenize],
                [clean, _remove_ttags, _cnndm_target_sentence_tokenization],
                nltk.word_tokenize,
                top_n,
            ),
            Summarization(
                test_source_file,
                test_target_file,
                [clean, tokenize.sent_tokenize],
                [clean, _remove_ttags, _cnndm_target_sentence_tokenization],
                nltk.word_tokenize,
                top_n,
            ),
        )

    return _setup_datasets(*((urls[0],) + args), **kwargs)


class CNNDMBertSumProcessedData:
    @staticmethod
    def save_data(data_iter, is_test=False, save_path="./", chunk_size=None):
        os.makedirs(save_path, exist_ok=True)

        def chunks(iterable, chunk_size):
            iterator = filter(None, iterable)  # iter(iterable)
            for first in iterator:
                if chunk_size:
                    yield itertools.chain([first], itertools.islice(iterator, chunk_size - 1))
                else:
                    yield itertools.chain([first], itertools.islice(iterator, None))

        chunks = chunks(data_iter, chunk_size)
        filename_list = []
        for i, chunked_data in enumerate(chunks):
            filename = f"{i}_test" if is_test else f"{i}_train"
            torch.save(list(chunked_data), os.path.join(save_path, filename))
            filename_list.append(os.path.join(save_path, filename))
        return filename_list

    @staticmethod
    def download(local_path=".data"):
        file_name = "bertsum_data.zip"
        url = "https://drive.google.com/uc?export=download&id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6"
        try:
            if os.path.exists(os.path.join(local_path, file_name)):
                zip = zipfile.ZipFile(os.path.join(local_path, file_name))
            else:
                dataset_zip = download_from_url(url, root=local_path)
                zip = zipfile.ZipFile(dataset_zip)
        except:
            print("Unexpected dataset downloading or reading error:", sys.exc_info()[0])
            raise

        zip.extractall(local_path)
        return local_path

    @classmethod
    def splits(cls, root):
        train_files = []
        test_files = []
        files = [join(root, f) for f in os.listdir(root) if isfile(join(root, f))]
        for fname in files:
            if fname.find("train") != -1:
                train_files.append(fname)
            elif fname.find("test") != -1:
                test_files.append(fname)

        def get_train_dataset():
            return get_dataset(train_files, True)

        def get_test_dataset():
            return get_dataset(test_files)

        return get_train_dataset, get_test_dataset
        # return get_cycled_dataset(get_dataset(train_files)), get_dataset(test_files)
