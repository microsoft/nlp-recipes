# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shutil
import time
import tempfile

from pyrouge import Rouge155
from rouge import Rouge
from .rouge_ext import RougeExt


def compute_rouge_perl(cand, ref, is_input_files=False, verbose=False):
    """
    Computes ROUGE scores using the python wrapper
    (https://github.com/bheinzerling/pyrouge) of perl ROUGE package.

    Args:
        cand (list or str): If `is_input_files` is `False`, `cand` is a list of strings
            containing predicted summaries. if `is_input_files` is `True`, `cand` is the path
            to the file containing the predicted summaries.
        ref (list or str): If `is_input_files` is `False`, `cand` is a list of strings
            containing reference summaries. if `is_input_files` is `True`, `cand` is the path
            to the file containing the reference summaries.
        is_input_files (bool, optional): If True, inputs are file names. Otherwise, inputs are lists
            of predicted and reference summaries. Defaults to False.
        verbose (bool, optional): If True, print out all rouge scores. Defaults to False.

    Returns:
        dict: Dictionary of ROUGE scores.

    """

    temp_dir = tempfile.mkdtemp()

    if is_input_files:
        candidates = [line.strip() for line in open(cand, encoding="utf-8")]
        references = [line.strip() for line in open(ref, encoding="utf-8")]
    else:
        candidates = cand
        references = ref

    print("Number of candidates: {}".format(len(candidates)))
    print("Number of references: {}".format(len(references)))
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))

    tmp_dir_candidate = tmp_dir + "/candidate/"
    tmp_dir_reference = tmp_dir + "/reference/"

    os.makedirs(tmp_dir_candidate, exist_ok=True)
    os.makedirs(tmp_dir_reference, exist_ok=True)

    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir_candidate + "/cand.{}.txt".format(i), "w", encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir_reference + "/ref.{}.txt".format(i), "w", encoding="utf-8") as f:
                f.write(references[i])
        r = Rouge155()
        r.model_dir = tmp_dir_reference
        r.system_dir = tmp_dir_candidate
        r.model_filename_pattern = "ref.#ID#.txt"
        r.system_filename_pattern = r"cand.(\d+).txt"
        rouge_results = r.convert_and_evaluate()
        if verbose:
            print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def compute_rouge_python(cand, ref, is_input_files=False, language="en"):
    """
    Computes ROUGE scores using the python package (https://pypi.org/project/py-rouge/).

    Args:
        cand (list or str): If `is_input_files` is `False`, `cand` is a list of strings
            containing predicted summaries. if `is_input_files` is `True`, `cand` is the path
            to the file containing the predicted summaries.
        ref (list or str): If `is_input_files` is `False`, `cand` is a list of strings
            containing reference summaries. if `is_input_files` is `True`, `cand` is the path
            to the file containing the reference summaries.
        is_input_files (bool, optional): If True, inputs are file names. Otherwise, inputs are
            lists of predicted and reference summaries. Defaults to False.
        language (str, optional): Language of the input text. Supported values are "en" and
            "hi". Defaults to "en".

    Returns:
        dict: Dictionary of ROUGE scores.

    """
    supported_langauges = ["en", "hi"]
    if language not in supported_langauges:
        raise Exception(
            "Language {0} is not supported. Supported languages are: {1}.".format(
                language, supported_langauges
            )
        )

    if is_input_files:
        candidates = [line.strip() for line in open(cand, encoding="utf-8")]
        references = [line.strip() for line in open(ref, encoding="utf-8")]
    else:
        candidates = cand
        references = ref

    print("Number of candidates: {}".format(len(candidates)))
    print("Number of references: {}".format(len(references)))
    assert len(candidates) == len(references)

    if language == "en":
        evaluator = Rouge(
            metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True
        )
    else:
        evaluator = RougeExt(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=False,
            apply_avg=True,
            language=language,
        )

    scores = evaluator.get_scores(candidates, [[it] for it in references])

    return scores
