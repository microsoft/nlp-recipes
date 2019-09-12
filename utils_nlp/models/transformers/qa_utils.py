# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright © Microsoft Corporation

import collections
import json
import math
import logging
import re
import string
import os
import jsonlines

logger = logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

from pytorch_transformers.tokenization_bert import BasicTokenizer
from pytorch_transformers import BertTokenizer, XLNetTokenizer
from pytorch_transformers.tokenization_bert import whitespace_tokenize


# from utils_nlp.models.transformers.common import MAX_SEQ_LEN

MAX_SEQ_LEN = 512
TOKENIZER_CLASSES = {"bert": BertTokenizer, "xlnet": XLNetTokenizer}


def _is_iterable_but_not_string(obj):
    """Check whether obj is a non-string Iterable."""
    return isinstance(obj, collections.Iterable) and not isinstance(obj, str)


QAInput = collections.namedtuple(
    "QAInput",
    ["doc_text", "question_text", "qa_id", "is_impossible", "answer_start", "answer_text"],
)


class QADataset(Dataset):
    def __init__(
        self,
        df,
        doc_text_col,
        question_text_col,
        qa_id_col,
        is_impossible_col=None,
        answer_start_col=None,
        answer_text_col=None,
    ):

        self.df = df.copy()
        self.doc_text_col = doc_text_col
        self.question_text_col = question_text_col

        ## TODO: can this be made optional???
        self.qa_id_col = qa_id_col

        if is_impossible_col is None:
            self.is_impossible_col = "is_impossible"
            df[self.is_impossible_col] = False
        else:
            self.is_impossible_col = is_impossible_col

        if answer_start_col is not None and answer_text_col is not None:
            self.actual_answer_available = True
        else:
            self.actual_answer_available = False
        self.answer_start_col = answer_start_col
        self.answer_text_col = answer_text_col

    def __getitem__(self, idx):
        current_item = self.df.iloc[idx, ]
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


def qa_data_generator(
    qa_dataset,
    model_type,
    sub_model_type,
    is_training,
    num_epochs=1,
    batch_size=32,
    to_lower=False,
    distributed=False,
    max_question_length=64,
    max_seq_len=MAX_SEQ_LEN,
    doc_stride=128,
    cache_dir="./cached_qa_features",
):

    if not os.path.exists(cached_qa_features):
        os.makedirs(cached_dir)

    tokenizer_class = TOKENIZER_CLASSES[model_type.value]
    tokenizer = tokenizer_class.from_pretrained(
        sub_model_type.value, do_lower_case=to_lower, cache_dir=cache_dir
    )

    sampler = DistributedSampler(qa_dataset) if distributed else RandomSampler(qa_dataset)
    data_loader = DataLoader(qa_dataset, sampler=sampler, batch_size=batch_size)

    if is_training and not qa_dataset.actual_answer_available:
        raise Exception("answer_start and answer_text must be provided for training data.")

    def _is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # We first project character-based annotations to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare,
        # but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index
    if is_training:
        examples_file = open(os.path.join(cache_dir, "cached_examples_train.jsonl"), "a+")
        features_file =  open(os.path.join(cache_dir, "cached_features_train.jsonl"), "a+")
    else:
        examples_file = open(os.path.join(cache_dir, "cached_examples_test.jsonl"), "a+")
        features_file =  open(os.path.join(cache_dir, "cached_features_test.jsonl"), "a+")
    examples_writer = jsonlines.Writer(examples_file)
    features_writer = jsonlines.Writer(features_file)

    features = []
    unique_id_all = []
    for epoch in num_epochs:
        for batch in data_loader:
            qa_examples = []
            qa_examples_json = []

            features_json = []

            for d_text, q_text, a_start, a_text, q_id, impossible in zip(
                batch.doc_text,
                batch.question_text,
                batch.answer_start,
                batch.answer_text,
                batch.qa_id,
                batch.is_impossible,
            ):
                a_start = a_start.item()
                d_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in d_text:
                    if _is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            d_tokens.append(c)
                        else:
                            d_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(d_tokens) - 1)

                if _is_iterable_but_not_string(a_start):
                    if len(a_start) != 1 and is_training and not impossible:
                        raise Exception("For training, each question should have exactly 1 answer.")
                    a_start = a_start[0]
                    a_text = a_text[0]

                start_position = None
                end_position = None
                if is_training:
                    if not impossible:
                        answer_length = len(a_text)
                        start_position = char_to_word_offset[a_start]
                        end_position = char_to_word_offset[a_start + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(d_tokens[start_position : (end_position + 1)])
                        cleaned_answer_text = " ".join(whitespace_tokenize(a_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning(
                                "Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text
                            )
                            continue
                    else:
                        start_position = -1
                        end_position = -1

                qa_examples.append(
                    QAExample(
                        qa_id=q_id,
                        doc_tokens=d_tokens,
                        question_text=q_text,
                        orig_answer_text=a_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=impossible,
                    )
                )
                if epoch == 1:
                    qa_examples_json.append({"qa_id": q_id, "doc_tokens": d_tokens})

            cls_token = "[CLS]"
            sep_token = "[SEP]"
            pad_token = 0
            sequence_a_segment_id = 0
            sequence_b_segment_id = 1
            cls_token_segment_id = 0
            pad_token_segment_id = 0
            cls_token_at_end = False
            mask_padding_with_zero = True

            # unique_id identified unique feature/label pairs. It's different
            # from qa_id in that each qa_example can be broken down into
            # multiple feature samples if the paragraph length is longer than
            # maximum sequence length allowed
            unique_id = 1000000000
            for (example_index, example) in enumerate(qa_examples):
                query_tokens = tokenizer.tokenize(example.question_text)

                if len(query_tokens) > max_question_length:
                    query_tokens = query_tokens[0:max_question_length]
                # map word-piece tokens to original tokens
                tok_to_orig_index = []
                # map original tokens to corresponding word-piece tokens
                orig_to_tok_index = []
                all_doc_tokens = []
                for (i, token) in enumerate(example.doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    sub_tokens = tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)

                tok_start_position = None
                tok_end_position = None
                if is_training and example.is_impossible:
                    tok_start_position = -1
                    tok_end_position = -1
                if is_training and not example.is_impossible:
                    tok_start_position = orig_to_tok_index[example.start_position]
                    if example.end_position < len(example.doc_tokens) - 1:
                        # +1: move the the token after the ending token in
                        # original tokens
                        # -1, moves one step back
                        # these two operations ensures word piece is covered
                        # when it's part of the original ending token.
                        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                    else:
                        tok_end_position = len(all_doc_tokens) - 1
                    (tok_start_position, tok_end_position) = _improve_answer_span(
                        all_doc_tokens,
                        tok_start_position,
                        tok_end_position,
                        tokenizer,
                        example.orig_answer_text,
                    )

                # The -3 accounts for [CLS], [SEP] and [SEP]
                max_tokens_for_doc = max_seq_len - len(query_tokens) - 3

                # We can have documents that are longer than the maximum sequence length.
                # To deal with this we do a sliding window approach, where we take chunks
                # of the up to our max length with a stride of `doc_stride`.
                _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
                doc_spans = []
                start_offset = 0
                while start_offset < len(all_doc_tokens):
                    length = len(all_doc_tokens) - start_offset
                    if length > max_tokens_for_doc:
                        length = max_tokens_for_doc
                    doc_spans.append(_DocSpan(start=start_offset, length=length))
                    if start_offset + length == len(all_doc_tokens):
                        break
                    start_offset += min(length, doc_stride)

                for (doc_span_index, doc_span) in enumerate(doc_spans):
                    tokens = []
                    token_to_orig_map = {}
                    token_is_max_context = {}
                    segment_ids = []

                    # p_mask: mask with 1 for token than cannot be in the answer
                    # (0 for token which can be in an answer)
                    # Original TF implem also keep the classification token (set to 0) (not sure why...)
                    ## TODO: Should we set p_mask = 1 for cls token?
                    p_mask = []

                    # CLS token at the beginning
                    if not cls_token_at_end:
                        tokens.append(cls_token)
                        segment_ids.append(cls_token_segment_id)
                        p_mask.append(0)
                        cls_index = 0

                    # Query
                    tokens += query_tokens
                    segment_ids += [sequence_a_segment_id] * len(query_tokens)
                    p_mask += [1] * len(query_tokens)

                    # SEP token
                    tokens.append(sep_token)
                    segment_ids.append(sequence_a_segment_id)
                    p_mask.append(1)

                    # Paragraph
                    for i in range(doc_span.length):
                        split_token_index = doc_span.start + i
                        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                        ## TODO: maybe this can be improved to compute
                        # is_max_context for each token only once.
                        is_max_context = _check_is_max_context(
                            doc_spans, doc_span_index, split_token_index
                        )
                        token_is_max_context[len(tokens)] = is_max_context
                        tokens.append(all_doc_tokens[split_token_index])
                        segment_ids.append(sequence_b_segment_id)
                        p_mask.append(0)

                    # SEP token
                    tokens.append(sep_token)
                    segment_ids.append(sequence_b_segment_id)
                    p_mask.append(1)

                    # CLS token at the end
                    if cls_token_at_end:
                        tokens.append(cls_token)
                        segment_ids.append(cls_token_segment_id)
                        p_mask.append(0)
                        cls_index = len(tokens) - 1  # Index of classification token

                    input_ids = tokenizer.convert_tokens_to_ids(tokens)

                    # The mask has 1 for real tokens and 0 for padding tokens. Only real
                    # tokens are attended to.
                    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

                    # Zero-pad up to the sequence length.
                    if len(input_ids) < max_seq_len:
                        pad_token_length = max_seq_len - len(input_ids)
                        pad_mask = 0 if mask_padding_with_zero else 1
                        input_ids += [pad_token] * pad_token_length
                        input_mask += [pad_mask] * pad_token_length
                        segment_ids += [pad_token_segment_id] * pad_token_length
                        p_mask += [1] * pad_token_length

                    assert len(input_ids) == max_seq_len
                    assert len(input_mask) == max_seq_len
                    assert len(segment_ids) == max_seq_len

                    span_is_impossible = example.is_impossible
                    start_position = None
                    end_position = None
                    if is_training and not span_is_impossible:
                        # For training, if our document chunk does not contain an annotation
                        # we throw it out, since there is nothing to predict.
                        doc_start = doc_span.start
                        doc_end = doc_span.start + doc_span.length - 1
                        out_of_span = False
                        if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                            out_of_span = True
                        if out_of_span:
                            start_position = 0
                            end_position = 0
                            span_is_impossible = True
                        else:
                            # +1 for [CLS] token
                            # +1 for [SEP] toekn
                            doc_offset = len(query_tokens) + 2
                            start_position = tok_start_position - doc_start + doc_offset
                            end_position = tok_end_position - doc_start + doc_offset

                    if is_training and span_is_impossible:
                        start_position = cls_index
                        end_position = cls_index

                    features.append(
                        QAFeatures(
                            unique_id=unique_id,
                            qa_id=example.qa_id,
                            tokens=tokens,
                            token_to_orig_map=token_to_orig_map,
                            token_is_max_context=token_is_max_context,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            start_position=start_position,
                            end_position=end_position,
                            cls_index=cls_index,
                            p_mask=p_mask,
                        )
                    )

                    unique_id += 1
                    unique_id_all.append(unique_id)
                    if epoch == 1:
                        features_json.append({"unique_id": unique_id,
                                            "tokens": tokens,
                                            "token_to_orign_map": token_to_orig_map,
                                            "token_is_max_context": token_is_max_context})
            if epoch == 1:
                examples_writer.write_all(qa_examples_json)
                features_writer.write_all(features_json)

            while len(features) >= batch_size:
                features_batch = features[:batch_size]
                unique_id_batch = unique_id[:batch_size]

                features = features[batch_size:]
                unique_id_all = unique_id_all[batch_size:]

                input_ids_batch = torch.tensor([f.input_ids for f in features_batch], dtype=torch.long)
                input_mask_batch = torch.tensor([f.input_mask for f in features_batch], dtype=torch.long)
                segment_ids_batch = torch.tensor([f.segment_ids for f in features_batch], dtype=torch.long)
                cls_index_batch = torch.tensor([f.cls_index for f in features_batch], dtype=torch.long)
                p_mask_batch = torch.tensor([f.p_mask for f in features_batch], dtype=torch.float)

                if is_training:
                    start_position_batch = torch.tensor(
                        [f.start_position for f in features_batch], dtype=torch.long
                    )
                    end_position_batch = torch.tensor(
                        [f.end_position for f in features_batch], dtype=torch.long
                    )
                    yield (
                        input_ids_batch_batch,
                        input_mask_batch,
                        segment_ids_batch,
                        start_position_batch,
                        end_position_batch,
                        cls_index_batch,
                        p_mask_batch,
                    )
                else:
                    yield (input_ids_batch, input_mask_batch, segment_ids_batch, cls_index_batch, p_mask_batch, unique_id_batch)


QAExample_ = collections.namedtuple(
    "QAExample",
    [
        "qa_id",
        "doc_tokens",
        "question_text",
        "orig_answer_text",
        "start_position",
        "end_position",
        "is_impossible",
    ],
)


# create a wrapper class so that we can add docstrings
class QAExample(QAExample_):
    """
    A data structure representing an unique document-question-answer triplet.

    Args:
        qa_id (int): An unique id identifying the document-question pair.
            This is used to map prediction results to ground truth answers
            during evaluation, because the data order is not preserved
            during pre-processing and post-processing.
        doc_tokens (list): White-space tokenized tokens of the document
            text. This is used to generate the final answer based on
            predicted start and end token indices during post-processing.
        question_text (str): Text of the question.
        orig_answer_text (str): Text of the ground truth answer if available.
        start_position (int): Index of the starting token of the answer
            span, if available.
        end_position (int): Index of the ending token of the answer span,
            if available.
        is_impossible (bool): If the question is impossible to answer based
            on the given document.
    """

    pass


QAFeatures_ = collections.namedtuple(
    "QAFeatures",
    [
        "unique_id",
        "qa_id",
        "tokens",
        "token_to_orig_map",
        "token_is_max_context",
        "input_ids",
        "input_mask",
        "segment_ids",
        "start_position",
        "end_position",
        "cls_index",
        "p_mask",
    ],
)


# create a wrapper class so that we can add docstrings
class QAFeatures(QAFeatures_):
    """
    BERT-format features of an unique document span-question-answer triplet.

    Args:
        unique_id (int): An unique id identifying the
            document-question-answer triplet.
        example_index (int): Index of the unique QAExample from which this
            feature instance is derived from. A single QAExample can derive
            multiple QAFeatures if the document is too long and gets split
            into multiple document spans. This index is used to group
            QAResults belonging to the same document-question pair and
            generate the final answer.
        tokens (list): Concatenated question tokens and paragraph tokens.
        token_to_orig_map (dict): A dictionary mapping token indices in the
            document span back to the token indices in the original document
            before document splitting.
            This is needed during post-processing to generate the final
            predicted answer.
        token_is_max_context (list): List of booleans indicating whether a
            token has the maximum context in teh current document span if it
            appears in multiple document spans and gets multiple predicted
            scores. We only want to consider the score with "maximum context".
            "Maximum context" is defined as the *minimum* of its left and
            right context.
            For example:
                Doc: the man went to the store and bought a gallon of milk
                Span A: the man went to the
                Span B: to the store and bought
                Span C: and bought a gallon of

            In the example the maximum context for 'bought' would be span C
            since it has 1 left context and 3 right context, while span B
            has 4 left context and 0 right context.
            This is needed during post-processing to generate the final
            predicted answer.
        input_ids (list): List of numerical token indices corresponding to
            the tokens.
        input_mask (list): List of 1s and 0s indicating if a token is from
            the input data or padded to conform to the maximum sequence
            length. 1 for actual token and 0 for padded token.
        segment_ids (list): List of 0s and 1s indicating if a token is from
            the question text (0) or paragraph text (1).
        start_position (int): Index of the starting token of the answer span.
        end_position (int): Index of the ending token of the answer span.

    """

    pass


QAResult_ = collections.namedtuple("QAResult", ["unique_id", "start_logits", "end_logits"])
QAResultExtended = collections.namedtuple(
    "QAResultExtended",
    [
        "unique_id",
        "start_top_log_probs",
        "start_top_index",
        "end_top_log_probs",
        "end_top_index",
        "cls_logits",
    ],
)


# create a wrapper class so that we can add docstrings
class QAResult(QAResult_):
    """
    Question answering prediction result returned by BERTQAExtractor.predict.

    Args:
        unique_id: Unique id identifying the training/testing sample. It's
            used to map the prediction result back to the QAFeatures
            during postprocessing.
        start_logits (list): List of logits for predicting each token being
            the start of the answer span.
        end__logits (list): List of logits for predicting each token being
            the end of the answer span.

    """

    pass


def postprocess_answer(
    results,
    examples,
    features,
    do_lower_case,
    n_best_size=20,
    max_answer_length=30,
    output_prediction_file="./qa_predictions.json",
    output_nbest_file="./nbest_predictions.json",
    unanswerable_exists=False,
    output_null_log_odds_file="./null_odds.json",
    null_score_diff_threshold=0.0,
    verbose_logging=False,
):
    """
    Postprocesses start and end logits generated by
    :meth:`utils_nlp.models.bert.BERTQAExtractor.fit`.

    Args:
        results (list): List of QAResults, each QAResult contains an
            unique id, answer start logits, and answer end logits. See
            :class:`.QAResult` for more details.
        examples (list): List of QAExamples. QAExample contains the original
            document tokens that are used to generate the final predicted
            answer from the predicted the start and end positions. See
            :class:`.QAExample` for more details.
        features (list): List of QAFeatures. QAFeatures contains the mapping
            from indices in the processed token list to the original
            document tokens that are used to generate the final predicted
            answers. See :class:`.QAFeatures` for more details.
        do_lower_case (bool): Whether an uncased tokenizer was used during
            data preprocessing. This is required during answer finalization
            by comparing the predicted answer text and the original text
            span in :func:`_get_final_text`.
        n_best_size (int, optional): The number of candidates to choose from
            each QAResult to generate the final prediction. It's also the
            maximum number of n-best answers to output for each question.
            Note that the number of n-best answers can be smaller than
            `n_best_size` because some unqualified answers, e.g. answer that
            are too long, are removed.
        max_answer_length (int, optional): Maximum length of the answer.
            Defaults to 30.
        output_prediction_file (str, optional): Path of the file to save the
            predicted answers. Defaults to "./qa_predictions.json".
        output_nbest_file (str, optional): Path of the file to save the
            n-best answers. Defaults to "./nbest_predictions.json".
        unanswerable_exists (bool, optional): Whether there are unanswerable
            questions in the data. If True, the start and end logits of the
            [CLS] token, which indicate the probability of the answer being
            empty, are included in the candidate answer list.  Defaults to
            False.
        output_null_log_odds_file (str, optional): If unanswerable_exists is
            True, the score difference between empty prediction and best
            non-empty prediction are saved to this file. These scores can be
            used to find the best threshold for predicting an empty answer.
            Defaults to "./null_odds.json".
        null_score_diff_threshold (float, optional): If the score
            difference between empty prediction and best non-empty
            prediction is higher than this threshold, the final predicted
            answer is empty. Defaults to 0.0.
        verbose_logging (bool, optional): Whether to log details of answer
            postprocessing. Defaults to False.

    Returns:
        tuple: (OrderedDict, OrderedDict, OrderedDict)
            The keys of the dictionaries are the `qa_id` of the original
            :class:`.QAExample` the answers correspond to.
            The values of the first dictionary are the predicted answer
            texts in string type.
            The values of the second dictionary  are the softmax
            probabilities of the predicted answers.
            The values of the third dictionary are the n-best answers for
            each qa_id. Note that the number of n-best answers can be smaller
            than `n_best_size` because some unqualified answers,
            e.g. answers that are too long, are removed.

    """
    # Helper functions
    def _get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heuristic between
        # `pred_text` and `orig_text` to get a character-to-character alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if verbose_logging:
                logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if verbose_logging:
                logger.info(
                    "Length not equal after stripping spaces: '%s' vs '%s'",
                    orig_ns_text,
                    tok_ns_text,
                )
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if verbose_logging:
                logger.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if verbose_logging:
                logger.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position : (orig_end_position + 1)]
        return output_text

    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def _compute_softmax(scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    # Helper functions end

    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    # example_index_to_features = collections.defaultdict(list)
    qa_id_to_features = collections.defaultdict(list)
    # Map unique features to the original doc-question-answer triplet
    # Each doc-question-answer triplet can have multiple features because the doc
    # could be split into multiple spans
    for f in features:
        # example_index_to_features[feature.example_index].append(feature)
        qa_id_to_features[f.qa_id].append(f)

    unique_id_to_result = {}
    for r in results:
        unique_id_to_result[r.unique_id] = r

    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
    )

    all_predictions = collections.OrderedDict()
    all_probs = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(examples):
        # get all the features belonging to the same example,
        # i.e. paragaraph/question pair.
        features = qa_id_to_features[example.qa_id]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if unanswerable_exists:
                # The first element of the start end end logits is the
                # probability of predicting the [CLS] token as the start and
                # end positions of the answer, which means the answer is
                # empty.
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if unanswerable_exists:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )

        # Sort by the sum of the start and end logits in ascending order,
        # so that the first element is the most probable answer
        prelim_predictions = sorted(
            prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True
        )

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = _get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit
                )
            )
        # if we didn't include the empty option in the n-best, include it
        if unanswerable_exists:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit, end_logit=null_end_logit
                    )
                )

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for ie, entry in enumerate(nbest):
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
                    best_non_null_entry_index = ie

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

            if entry.text == "":
                null_prediction_index = i

        assert len(nbest_json) >= 1

        if not unanswerable_exists:
            all_predictions[example.qa_id] = nbest_json[0]["text"]
            all_probs[example.qa_id] = nbest_json[0]["probability"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (
                score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            )
            scores_diff_json[example.qa_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qa_id] = ""
                ## TODO: double check this
                all_probs[example.qa_id] = probs[null_prediction_index]
            else:
                all_predictions[example.qa_id] = best_non_null_entry.text
                all_probs[example.qa_id] = probs[best_non_null_entry_index]
        all_nbest_json[example.qa_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if unanswerable_exists:
        logger.info("Writing null odds to: %s" % (output_null_log_odds_file))
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, all_probs, all_nbest_json


def evaluate_qa(qa_ids, actuals, preds, na_probs=None, na_prob_thresh=0, out_file=None):
    """
    Evaluate question answering prediction results against ground truth answers.

    Args:
        qa_ids (list): List of ids identifying unique document-question pairs.
        actuals (list): List of ground truth answer texts corresponding the
            qa_ids.
        preds (dict): Dictionary of qa_id and predicted answer pairs. This
            is a dictionary because the data order is not preserved during
            pre- and post-processing.
        na_probs (dict, optional): Dictionary of qa_id and unanswerable
            probability pairs. If None, unanswerable probabilities are all
            set to zero. Defaults to None.
        na_prob_thresh (float, optional): Probability threshold to predict a
            question to be unanswerable. If `na_probs` > `na_prob_thresh`,
            the prediction is considered as correct if the question
            is unanswerable. Otherwise, the prediction is considered as
            incorrect. Defaults to 0.
        out_file (str, optional): Path of the file to save the evaluation
            results to. Defaults to None.
    """

    # Helper functions
    def _normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _get_tokens(s):
        """Normalizes text and returns white-space tokenized tokens. """
        if not s:
            return []
        return _normalize_answer(s).split()

    def _compute_exact(a_gold, a_pred):
        """Compute the exact match between two sentences after normalization.

        Returns:
            int: 1 if two sentences match exactly after normalization,
                0 otherwise.
        """
        return int(_normalize_answer(a_gold) == _normalize_answer(a_pred))

    def _compute_f1(a_gold, a_pred):
        """
            Compute F1 score based on token overlapping between two
            sentences.
        """
        gold_toks = _get_tokens(a_gold)
        pred_toks = _get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _get_raw_scores(qa_ids, actuals, preds):
        """
            Compute exact match and F1 scores without applying any
            unanswerable probability threshold.
        """
        exact_scores = {}
        f1_scores = {}

        for qid, gold_answers in zip(qa_ids, actuals):
            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = [""]
            if qid not in preds:
                print("Missing prediction for %s" % qid)
                continue
            a_pred = preds[qid]
            # Take max over all gold answers
            exact_scores[qid] = max(_compute_exact(a, a_pred) for a in gold_answers)
            f1_scores[qid] = max(_compute_f1(a, a_pred) for a in gold_answers)
        return exact_scores, f1_scores

    def _apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
        """Update the input scores by applying unanswerable probability threshold."""

        new_scores = {}
        for qid, s in scores.items():
            pred_na = na_probs[qid] > na_prob_thresh
            if pred_na:
                new_scores[qid] = float(not qid_to_has_ans[qid])
            else:
                new_scores[qid] = s
        return new_scores

    def _make_eval_dict(exact_scores, f1_scores, qid_list=None):
        """Create a dictionary of evaluation results."""
        if not qid_list:
            total = len(exact_scores)
            return collections.OrderedDict(
                [
                    ("exact", 100.0 * sum(exact_scores.values()) / total),
                    ("f1", 100.0 * sum(f1_scores.values()) / total),
                    ("total", total),
                ]
            )
        else:
            total = len(qid_list)
            return collections.OrderedDict(
                [
                    ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                    ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                    ("total", total),
                ]
            )

    def _merge_eval(main_eval, new_eval, prefix):
        """Merge multiple evaluation result dictionaries."""
        for k in new_eval:
            main_eval["%s_%s" % (prefix, k)] = new_eval[k]

    # Helper functions end

    if na_probs is None:
        na_probs = {k: 0.0 for k in preds}

    qid_to_has_ans = {qa_id: bool(ans) for (qa_id, ans) in zip(qa_ids, actuals)}
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = _get_raw_scores(qa_ids, actuals, preds)
    exact_thresh = _apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, na_prob_thresh)
    f1_thresh = _apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, na_prob_thresh)
    out_eval = _make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = _make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
        _merge_eval(out_eval, has_ans_eval, "HasAns")
    if no_ans_qids:
        no_ans_eval = _make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
        _merge_eval(out_eval, no_ans_eval, "NoAns")

    if out_file:
        with open(out_file, "w") as f:
            json.dump(out_eval, f)
    else:
        print(json.dumps(out_eval, indent=2))
    return out_eval
