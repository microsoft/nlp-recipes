# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script reuses some code from https://github.com/nlpyang/BertSum

"""
    Utility functions for downloading, extracting, and reading the
    CNN/DM dataset at https://github.com/harvardnlp/sent-summary.
    
"""

import glob
import nltk

nltk.download("punkt")
from nltk import tokenize
import os
import sys
import regex as re
import torch
import torchtext
from torchtext.utils import download_from_url, extract_archive
import zipfile


from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.models.transformers.datasets import SummarizationDataset
from utils_nlp.models.transformers.extractive_summarization import get_dataset, get_dataloader





def CNNDMSummarizationDataset(*args, **kwargs):
    """Load the CNN/Daily Mail dataset preprocessed by harvardnlp group."""

    REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}

    
    def _clean(x):
        return re.sub(
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
            lambda m: REMAP.get(m.group()), x)


    def _remove_ttags(line):
        line = re.sub(r"<t>", "", line)
        # change </t> to <q>
        # pyrouge test requires <q> as sentence splitter
        line = re.sub(r"</t>", "<q>", line)
        return line


    def _target_sentence_tokenization(line):
        return line.split("<q>")

    URLS = ["https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz"]
    

    def _setup_datasets(url, top_n=-1, local_cache_path=".data"):
        FILE_NAME = "cnndm.tar.gz"
        maybe_download(url, FILE_NAME, local_cache_path)
        dataset_tar = os.path.join(local_cache_path, FILE_NAME)
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
            SummarizationDataset(
                train_source_file,
                train_target_file,
                [_clean, tokenize.sent_tokenize],
                [_clean, _remove_ttags, _target_sentence_tokenization],
                nltk.word_tokenize,
                top_n,
            ),
            SummarizationDataset(
                test_source_file,
                test_target_file,
                [_clean, tokenize.sent_tokenize],
                [_clean, _remove_ttags, _target_sentence_tokenization],
                nltk.word_tokenize,
                top_n,
            ),
        )

    return _setup_datasets(*((URLS[0],) + args), **kwargs)
    
    
class CNNDMBertSumProcessedData:
    """Class to load dataset preprocessed by BertSum paper at 
        https://github.com/nlpyang/BertSum
    """

    @staticmethod
    def download(local_path=".data"):
        file_name = "bertsum_data.zip"
        url = "https://drive.google.com/uc?export=download&id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6"
        try:
            if os.path.exists(os.path.join(local_path, file_name)):
                downloaded_zipfile = zipfile.ZipFile(os.path.join(local_path, file_name))
            else:
                dataset_zip = download_from_url(url, root=local_path)
                downloaded_zipfile = zipfile.ZipFile(dataset_zip)
        except:
            print("Unexpected dataset downloading or reading error:", sys.exc_info()[0])
            raise

        downloaded_zipfile.extractall(local_path)
        return local_path

    
