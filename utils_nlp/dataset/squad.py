# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import pandas as pd

from utils_nlp.dataset.url_utils import maybe_download

URL_DICT = {
    "v1.1": {"train": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json",
             "dev": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"},
    "v2.0": {"train": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v2.0.json",
             "dev": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json"}
}

def load_pandas_df(local_cache_path=".", squad_version="v1.1", file_split="train"):
    URL = URL_DICT[squad_version][file_split]
    file_name = URL.split("/")[-1]
    maybe_download(URL, file_name, local_cache_path)

    file_path = os.path.join(local_cache_path, file_name)

    with open(file_path, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    paragraph_text_list = []
    question_text_list = []
    answer_start_list = []
    answer_text_list = []
    qa_id_list = []
    is_impossible_list = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                answer_offset = None
                orig_answer_text = None
                is_impossible = False
                if file_split == "train":
                    if squad_version == "v2.0":
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                    else:
                        orig_answer_text = ""
            
            paragraph_text_list.append(paragraph_text)
            question_text_list.append(question_text)
            answer_start_list.append(answer_offset)
            answer_text_list.append(orig_answer_text)
            qa_id_list.append(qas_id)
            is_impossible_list.append(is_impossible)

    output_df = pd.DataFrame({"doc_text": paragraph_text_list, 
                              "question_text": question_text_list,
                              "answer_start": answer_start_list,
                              "answer_text": answer_text_list,
                              "qa_id": qa_id_list,
                              "is_impossible": is_impossible_list}
                              )

    return output_df