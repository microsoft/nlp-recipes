# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import collections
import itertools
import os
import torch
from torch.utils.data import Dataset, IterableDataset
from multiprocessing import Pool, cpu_count
from functools import partial
import jsonlines


class SCDataSet(Dataset):
    """Dataset for single sequence classification tasks"""

    def __init__(self, df, text_col, label_col, transform, **transform_args):
        self.df = df
        cols = list(df.columns)
        self.transform = transform
        self.transform_args = transform_args

        if isinstance(text_col, int):
            self.text_col = text_col
        elif isinstance(text_col, str):
            self.text_col = cols.index(text_col)
        else:
            raise TypeError("text_col must be of type int or str")

        if label_col is None:
            self.label_col = None
        elif isinstance(label_col, int):
            self.label_col = label_col
        elif isinstance(label_col, str):
            self.label_col = cols.index(label_col)
        else:
            raise TypeError("label_col must be of type int or str")

    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids = self.transform(
            self.df.iloc[idx, self.text_col], **self.transform_args
        )
        if self.label_col is None:
            return tuple(
                [
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                ]
            )
        labels = self.df.iloc[idx, self.label_col]
        return tuple(
            [
                torch.tensor(input_ids, dtype=torch.long),  # input_ids
                torch.tensor(attention_mask, dtype=torch.long),  # attention_mask
                torch.tensor(token_type_ids, dtype=torch.long),  # segment ids
                torch.tensor(labels, dtype=torch.long),  # labels
            ]
        )

    def __len__(self):
        return self.df.shape[0]


class SPCDataSet(Dataset):
    """Dataset for sequence pair classification tasks"""

    def __init__(
        self, df, text1_col, text2_col, label_col, transform, **transform_args
    ):
        self.df = df
        cols = list(df.columns)
        self.transform = transform
        self.transform_args = transform_args

        if isinstance(text1_col, int):
            self.text1_col = text1_col
        elif isinstance(text1_col, str):
            self.text1_col = cols.index(text1_col)
        else:
            raise TypeError("text1_col must be of type int or str")

        if isinstance(text2_col, int):
            self.text2_col = text2_col
        elif isinstance(text2_col, str):
            self.text2_col = cols.index(text2_col)
        else:
            raise TypeError("text2_col must be of type int or str")

        if label_col is None:
            self.label_col = None
        elif isinstance(label_col, int):
            self.label_col = label_col
        elif isinstance(label_col, str):
            self.label_col = cols.index(label_col)
        else:
            raise TypeError("label_col must be of type int or str")

    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids = self.transform(
            self.df.iloc[idx, self.text1_col],
            self.df.iloc[idx, self.text2_col],
            **self.transform_args,
        )

        if self.label_col is None:
            return tuple(
                [
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                ]
            )

        labels = self.df.iloc[idx, self.label_col]
        return tuple(
            [
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ]
        )

    def __len__(self):
        return self.df.shape[0]


# QAInput is a data structure representing an unique document-question-answer triplet.
# Args:
#    doc_text (str): Input document text.
#    question_text(str): Input question text.
#    qa_id (int or str): An unique id identifying a document-question-answer sample.
#    is_impossible (bool): If the question is impossible to answer based on the input
#    document.
#    answer_start (int or list): Index of the answer start word in doc_text. For
#        testing data, this can be a list of integers for multiple ground truth answers.
#    answer_text (str or list): Text of the answer. For testing data, this can be a
#        list of strings
#        for multiple ground truth answers.
QAInput = collections.namedtuple(
    "QAInput",
    [
        "doc_text",
        "question_text",
        "qa_id",
        "is_impossible",
        "answer_start",
        "answer_text",
    ],
)


class QADataset(Dataset):
    def __init__(
        self,
        df,
        doc_text_col,
        question_text_col,
        qa_id_col=None,
        answer_start_col=None,
        answer_text_col=None,
        is_impossible_col=None,
    ):
        """
        A standard dataset structure for question answering that can be processed by
        :meth:`utils_nlp.models.transformers.question_answering.QAProcessor.preprocess`

        Args:
            df (pandas.DataFrame): Input data frame.
            doc_text_col (str): Name of the column containing the document texts.
            question_text_col (str): Name of the column containing the question texts.
            qa_id_col (str, optional): Name of the column containing the unique ids
                identifying document-question-answer samples. If not provided, a
                "qa_id" column is automatically created. Defaults to None.
            answer_start_col (str, optional): Name of the column containing answer
                start indices. For testing data, each value in the column can be a list
                of integers for multiple ground truth answers. Defaults to None.
            answer_text_col (str, optional): Name of the column containing answer texts.
                For testing data, each value in the column can be a list of strings for
                multiple ground truth answers. Defaults to None.
            is_impossible_col (str, optional): Name of the column containing boolean
                values indicating if the question is impossible to answer. If not
                provided, a "is_impossible" column is automatically created and
                populated with False. Defaults to None.
        """
        self.df = df.copy()
        self.doc_text_col = doc_text_col
        self.question_text_col = question_text_col

        if qa_id_col is None:
            self.qa_id_col = "qa_id"
            self.df[self.qa_id_col] = list(range(self.df.shape[0]))
        else:
            self.qa_id_col = qa_id_col

        if is_impossible_col is None:
            self.is_impossible_col = "is_impossible"
            self.df[self.is_impossible_col] = False
        else:
            self.is_impossible_col = is_impossible_col

        if answer_start_col is not None and answer_text_col is not None:
            self.actual_answer_available = True
        else:
            self.actual_answer_available = False
        self.answer_start_col = answer_start_col
        self.answer_text_col = answer_text_col

    def __getitem__(self, idx):
        current_item = self.df.iloc[
            idx,
        ]
        if self.actual_answer_available:
            return QAInput(
                doc_text=current_item[self.doc_text_col],
                question_text=current_item[self.question_text_col],
                qa_id=current_item[self.qa_id_col],
                is_impossible=current_item[self.is_impossible_col],
                answer_start=current_item[self.answer_start_col],
                answer_text=current_item[self.answer_text_col],
            )
        else:
            return QAInput(
                doc_text=current_item[self.doc_text_col],
                question_text=current_item[self.question_text_col],
                qa_id=current_item[self.qa_id_col],
                is_impossible=current_item[self.is_impossible_col],
                answer_start=-1,
                answer_text="",
            )

    def __len__(self):
        return self.df.shape[0]


def _line_iter(file_path):
    with open(file_path, "r", encoding="utf8") as fd:
        for line in fd:
            yield line


def _preprocess(sentences, preprocess_pipeline, word_tokenize=None):
    """
    Helper function to preprocess a list of paragraphs.

    Args:
        param (Tuple): params are tuple of (a list of strings,
            a list of preprocessing functions, and function to tokenize
            setences into words). A paragraph is represented with a
            single string with multiple setnences.

    Returns:
        list of list of strings, where each string is a token or word.
    """
    if preprocess_pipeline is not None:
        for function in preprocess_pipeline:
            sentences = function(sentences)

    if word_tokenize is None:
        return sentences
    else:
        return sentences, [word_tokenize(sentence) for sentence in sentences]


def _create_data_from_iterator(iterator, preprocessing, word_tokenize):
    for line in iterator:
        yield _preprocess(
            sentences=line,
            preprocess_pipeline=preprocessing,
            word_tokenize=word_tokenize,
        )


class IterableSummarizationDataset(IterableDataset):
    def __init__(
        self,
        source_file,
        target_file=None,
        source_preprocessing=None,
        target_preprocessing=None,
        word_tokenization=None,
        top_n=-1,
    ):
        """
        Create a summarization dataset instance given the
        paths of the source file and the target file

        Args:
            source_file (str): Full path of the file which contains a list of
                the paragraphs with line break as seperator.
            target_file (str): Full path of the file which contains a list of
                the summaries for the paragraphs in the source file with line break as
                seperator.
            source_preprocessing (list of functions): A list of preprocessing functions
                to process the paragraphs in the source file.
            target_preprocessing (list of functions): A list of preprocessing functions
                to process the paragraphs in the source file.
            word_tokenization (function): Tokenization function for tokenize the
                paragraphs and summaries. The tokenization method is used for sentence
                selection in
                :meth:`utils_nlp.models.transformers.extractive_summarization.
                ExtSumProcessor.preprocess`
            top_n (int, optional): The number which specifies how many examples in the
                beginning of the paragraph and summary lists that will be processed by
                this function. Defaults to -1, which means the whole lists of paragraphs
                and summaries should be procsssed.
        """

        source_iter = _line_iter(source_file)

        if top_n != -1:
            source_iter = itertools.islice(source_iter, top_n)

        self._source = _create_data_from_iterator(
            source_iter, source_preprocessing, word_tokenization
        )

        if target_file:
            target_iter = _line_iter(target_file)
            if top_n != -1:
                target_iter = itertools.islice(target_iter, top_n)
            self._target = _create_data_from_iterator(
                target_iter, target_preprocessing, word_tokenization
            )
        else:
            self._target = None

    def __iter__(self):
        for x in self._source:
            yield x

    def get_source(self):
        return self._source

    def get_target(self):
        return self._target


class SummarizationDataset(Dataset):
    def __init__(
        self,
        source_file,
        source=None,
        target_file=None,
        target=None,
        source_preprocessing=None,
        target_preprocessing=None,
        word_tokenize=None,
        top_n=-1,
        n_processes=-1,
    ):
        """
        Create a summarization dataset instance given the
        paths of the source file and the target file.

        Args:
            source_file (str): Full path of the file which contains a list of
                the input paragraphs with line break as seperator.
            source (list of str, optional): a list of input paragraphs.
                Defaults to None.
            target_file (str, optional): Full path of the file which contains a list of
                the summaries for the paragraphs in the source file with line break
                as seperator.
            target (list of str, optional): a list of summaries correponding to
                `source`. Defaults to None.
            source_preprocessing (list of functions): A list of preprocessing functions
                to process the paragraphs in the source file.
            target_preprocessing (list of functions): A list of preprocessing functions
                to process the summaries in the target file.
            top_n (int, optional): Number of examples to load from the input files.
                Defaults to -1, which means the entire dataset is loaded.
            n_processes (int, optional): Number of CPUs to use to process the data in
                parallel. Defaults to -1, which means all the CPUs will be used.
        """
        self._source_txt = []
        if source_file is not None and os.path.exists(source_file):
            with open(source_file, encoding="utf-8") as f:
                if top_n != -1:
                    self._source_txt = list(itertools.islice(f, top_n))
                else:
                    self._source_txt = f.readlines()
        if source:
            self._source_txt.extend(source)

        self._target_txt = []
        if target_file is not None and os.path.exists(target_file):
            with open(target_file, encoding="utf-8") as f:
                if top_n != -1:
                    self._target_txt = list(itertools.islice(f, top_n))
                else:
                    self._target_txt = f.readlines()
        if target:
            self._target_txt.extend(target)

        if len(self._target_txt) == 0:
            self._target_txt = None
        else:
            assert len(self._source_txt) == len(self._target_txt)

        result = parallel_preprocess(
            self._source_txt,
            preprocess_pipeline=source_preprocessing,
            word_tokenize=word_tokenize,
            num_pool=n_processes,
        )
        if word_tokenize:
            self._source_txt = list(
                map(lambda x: x[0], filter(lambda x: len(x[0]) > 0, result))
            )
            self._source = list(
                map(lambda x: x[1], filter(lambda x: len(x[1]) > 0, result))
            )
        else:
            self._source = list(
                map(lambda x: x, filter(lambda x: len(x) > 0, result))
            )

        if self._target_txt is not None and len(self._target_txt) > 0:
            result = parallel_preprocess(
                self._target_txt,
                preprocess_pipeline=target_preprocessing,
                word_tokenize=word_tokenize,
                num_pool=n_processes,
            )

            if word_tokenize:
                self._target_txt = list(
                    map(lambda x: x[0], filter(lambda x: len(x[0]) > 0, result))
                )
                self._target = list(
                    map(lambda x: x[1], filter(lambda x: len(x[1]) > 0, result))
                )
            else:
                self._target = list(
                    map(lambda x: x, filter(lambda x: len(x) > 0, result))
                )

    def shorten(self, top_n=None):
        if top_n is None:
            return self
        elif top_n <= len(self._source):
            self._source = self._source[0:top_n]
            self._source_txt = self._source_txt[0:top_n]

            if self._target_txt is not None:
                self._target = self._target[0:top_n]
                self._target_txt = self._target_txt[0:top_n]
            return self
        else:
            return self

    def __getitem__(self, idx):
        ## tupe is more adaptive
        if self._target_txt is None:
            return {"src": self._source[idx], "src_txt": self._source_txt[idx]}
        else:
            return {
                "src": self._source[idx],
                "src_txt": self._source_txt[idx],
                "tgt": self._target[idx],
                "tgt_txt": self._target_txt[idx],
            }

    def __len__(self):
        return len(self._source)

    def get_source(self):
        return self._source

    def get_source_txt(self):
        return self._source_txt

    def get_target_txt(self):
        return self._target_txt

    def get_target(self):
        return self._target

    def save_to_jsonl(self, output_file):
        with jsonlines.open(output_file, mode="w") as writer:
            if self._target_txt is None:
                for src in self._source:
                    writer.write({"src": src})
            else:
                for src, tgt in zip(self._source, self._target):
                    writer.write({"src": src, "tgt": tgt})


def parallel_preprocess(
    input_data, preprocess_pipeline, word_tokenize=None, num_pool=-1
):
    """
    Process data in parallel using multiple CPUs.

    Args:
        input_data (list): List if input strings to process.
        preprocess_pipeline (list): List of functions to apply on the input data.
        word_tokenize (func, optional): A tokenization function used to tokenize
            the results from preprocess_pipeline.
        num_pool (int, optional): Number of CPUs to use. Defaults to -1 and all
            available CPUs are used.

    Returns:
        list: list of processed text strings.

    """
    if num_pool == -1:
        num_pool = cpu_count()

    num_pool = min(num_pool, len(input_data))

    p = Pool(num_pool)

    results = p.map(
        partial(
            _preprocess,
            preprocess_pipeline=preprocess_pipeline,
            word_tokenize=word_tokenize,
        ),
        input_data,
        chunksize=min(1, int(len(input_data) / num_pool)),
    )
    p.close()
    p.join()

    return results
