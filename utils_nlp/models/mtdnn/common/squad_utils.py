import collections
import json
import math
import os
import string

import numpy as np
import six
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils_nlp.models.mtdnn.common.types import EncoderModelType

LARGE_NEG_NUM = -1.0e5
tokenizer = None


def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def calc_tokenized_span_range(
    context, question, answer, answer_start, answer_end, tokenizer, encoderModelType, verbose=False
):
    """
    :param context:
    :param question:
    :param answer:
    :param answer_start:
    :param answer_end:
    :param tokenizer:
    :param encoderModelType:
    :param verbose:
    :return: span_start, span_end
    """
    assert encoderModelType == EncoderModelType.BERT
    prefix = context[:answer_start]
    prefix_tokens = tokenizer.tokenize(prefix)
    full = context[:answer_end]
    full_tokens = tokenizer.tokenize(full)
    span_start = len(prefix_tokens)
    span_end = len(full_tokens)
    span_tokens = full_tokens[span_start:span_end]
    recovered_answer = " ".join(span_tokens).replace(" ##", "")
    cleaned_answer = " ".join(tokenizer.basic_tokenizer.tokenize(answer))
    if verbose:
        try:
            assert recovered_answer == cleaned_answer, (
                "answer: %s, recovered_answer: %s, question: %s, select:%s ext_select:%s context: %s"
                % (
                    cleaned_answer,
                    recovered_answer,
                    question,
                    context[answer_start:answer_end],
                    context[answer_start - 5 : answer_end + 5],
                    context,
                )
            )
        except Exception as e:
            pass
            print(e)
    return span_start, span_end


def is_valid_sample(context, answer_start, answer_end, answer):
    valid = True
    constructed = context[answer_start:answer_end]
    if constructed.lower() != answer.lower():
        valid = False
        return valid
    # check if it is inside of a token
    if answer_start > 0 and answer_end < len(context) - 1:
        prefix = context[answer_start - 1 : answer_start]
        suffix = context[answer_end : answer_end + 1]
        if len(remove_punc(prefix)) > 0 or len(remove_punc(suffix)):
            valid = False
    return valid


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def parse_squad_label(label):
    """
    :param label:
    :return: answer_start, answer_end, answer, is_impossible
    """
    answer_start, answer_end, is_impossible, answer = label.split(":::")
    answer_start = int(answer_start)
    answer_end = int(answer_end)
    is_impossible = int(is_impossible)
    return answer_start, answer_end, answer, is_impossible


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    # It is copyed from: https://github.com/google-research/bert/blob/master/run_squad.py
    # The SQuAD annotations are character based. We first project them to
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
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # It is copyed from: https://github.com/google-research/bert/blob/master/run_squad.py
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


def doc_split(doc_subwords, doc_stride=180, max_tokens_for_doc=384):
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(doc_subwords):
        length = len(doc_subwords) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(doc_subwords):
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def recompute_span(answer, answer_offset, char_to_word_offset):
    answer_length = len(answer)
    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + answer_length - 1]
    return start_position, end_position


def is_valid_answer(context, answer_start, answer_end, answer):
    valid = True
    constructed = " ".join(context[answer_start : answer_end + 1]).lower()
    cleaned_answer_text = " ".join(answer.split()).lower()
    if constructed.find(cleaned_answer_text) == -1:
        valid = False
    return valid


def token_doc(paragraph_text):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset


class InputFeatures(object):
    def __init__(
        self,
        unique_id,
        example_index,
        doc_span_index,
        tokens,
        token_to_orig_map,
        token_is_max_context,
        input_ids,
        input_mask,
        segment_ids,
        start_position=None,
        end_position=None,
        is_impossible=None,
        doc_offset=0,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.doc_offset = doc_offset

    def __str__(self):
        return json.dumps(
            {
                "unique_id": self.unique_id,
                "example_index": self.example_index,
                "doc_span_index": self.doc_span_index,
                "tokens": self.tokens,
                "token_to_orig_map": self.token_to_orig_map,
                "token_is_max_context": self.token_is_max_context,
                "input_ids": self.input_ids,
                "input_mask": self.input_mask,
                "segment_ids": self.segment_ids,
                "start_position": self.start_position,
                "end_position": self.end_position,
                "is_impossible": self.is_impossible,
                "doc_offset": self.doc_offset,
            }
        )


def mrc_feature(
    tokenizer,
    unique_id,
    example_index,
    query,
    doc_tokens,
    answer_start_adjusted,
    answer_end_adjusted,
    is_impossible,
    max_seq_len,
    max_query_len,
    doc_stride,
    answer_text=None,
    is_training=True,
):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    query_ids = tokenizer.tokenize(query)
    query_ids = query_ids[0:max_query_len] if len(query_ids) > max_query_len else query_ids
    max_tokens_for_doc = max_seq_len - len(query_ids) - 3
    unique_id_cp = unique_id
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    tok_start_position = None
    tok_end_position = None
    if is_training and is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not is_impossible:
        tok_start_position = orig_to_tok_index[answer_start_adjusted]
        if answer_end_adjusted < len(doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[answer_end_adjusted + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, answer_text
        )

    doc_spans = doc_split(
        all_doc_tokens, doc_stride=doc_stride, max_tokens_for_doc=max_tokens_for_doc
    )
    feature_list = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = ["[CLS]"] + query_ids + ["[SEP]"]
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = [0 for i in range(len(tokens))]

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        doc_offset = len(query_ids) + 2

        start_position = None
        end_position = None
        if is_training and not is_impossible:
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
            else:
                # doc_offset = len(query_ids) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        if is_training and is_impossible:
            start_position = 0
            end_position = 0
        is_impossible = True if is_impossible else False
        feature = InputFeatures(
            unique_id=unique_id_cp,
            example_index=example_index,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible,
            doc_offset=doc_offset,
        )
        feature_list.append(feature)
        unique_id_cp += 1
    return feature_list


def gen_gold_name(dir, path, version, suffix="json"):
    fname = "{}-{}.{}".format(path, version, suffix)
    return os.path.join(dir, fname)


def load_squad_label(path):
    rows = {}
    with open(path, encoding="utf8") as f:
        data = json.load(f)["data"]
    for article in tqdm.tqdm(data, total=len(data)):
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                uid, question = qa["id"], qa["question"]
                is_impossible = qa.get("is_impossible", False)
                label = 1 if is_impossible else 0
                rows[uid] = label
    return rows


def position_encoding(m, threshold=4):
    encoding = np.ones((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(i, m):
            if j - i > threshold:
                encoding[i][j] = float(1.0 / math.log(j - i + 1))
    return torch.from_numpy(encoding)


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
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
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
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
    # tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    global tokenizer
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def masking_score(mask, batch_meta, start, end, keep_first_token=False):
    """For MRC, e.g., SQuAD
    """
    start = start.data.cpu()
    end = end.data.cpu()
    score_mask = start.new(mask.size()).zero_()
    score_mask = score_mask.data.cpu()
    token_is_max_contexts = batch_meta["token_is_max_context"]
    doc_offsets = batch_meta["doc_offset"]
    word_maps = batch_meta["token_to_orig_map"]
    batch_size = score_mask.size(0)
    doc_len = score_mask.size(1)
    for i in range(batch_size):
        doc_offset = doc_offsets[i]
        if keep_first_token:
            score_mask[i][1:doc_offset] = 1.0
        else:
            score_mask[i][:doc_offset] = 1.0
        for j in range(doc_len):
            sj = str(j)
            if mask[i][j] == 0:
                score_mask[i][j] == 1.0
            if sj in token_is_max_contexts[i] and (not token_is_max_contexts[i][sj]):
                score_mask[i][j] == 1.0
    score_mask = score_mask * LARGE_NEG_NUM
    start = start + score_mask
    end = end + score_mask
    start = F.softmax(start, 1)
    end = F.softmax(end, 1)
    return start, end


def extract_answer(batch_meta, batch_data, start, end, keep_first_token=False, max_len=5):
    doc_len = start.size(1)
    pos_enc = position_encoding(doc_len, max_len)
    token_is_max_contexts = batch_meta["token_is_max_context"]
    doc_offsets = batch_meta["doc_offset"]
    word_maps = batch_meta["token_to_orig_map"]
    tokens = batch_meta["tokens"]
    contexts = batch_meta["doc"]
    uids = batch_meta["uids"]
    mask = batch_data[batch_meta["mask"]].data.cpu()
    # need to fill mask
    start, end = masking_score(mask, batch_meta, start, end)
    #####
    predictions = []
    answer_scores = []

    for i in range(start.size(0)):
        uid = uids[i]
        scores = torch.ger(start[i], end[i])
        scores = scores * pos_enc
        scores.triu_()
        scores = scores.numpy()
        best_idx = np.argpartition(scores, -1, axis=None)[-1]
        best_score = np.partition(scores, -1, axis=None)[-1]
        s_idx, e_idx = np.unravel_index(best_idx, scores.shape)
        s_idx, e_idx = int(s_idx), int(e_idx)
        ###
        tok_tokens = tokens[i][s_idx : (e_idx + 1)]
        tok_text = " ".join(tok_tokens)
        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")
        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        ###
        context = contexts[i].split()
        rs = word_maps[i][str(s_idx)]
        re = word_maps[i][str(e_idx)]
        raw_answer = " ".join(context[rs : re + 1])
        # extract final answer
        answer = get_final_text(tok_text, raw_answer, True, False)
        predictions.append(answer)
        answer_scores.append(float(best_score))
    return predictions, answer_scores


def select_answers(ids, predictions, scores):
    assert len(ids) == len(predictions)
    predictions_list = {}
    for idx, uid in enumerate(ids):
        score = scores[idx]
        ans = predictions[idx]
        lst = predictions_list.get(uid, [])
        lst.append((ans, score))
        predictions_list[uid] = lst
    final = {}
    scores = {}
    for key, val in predictions_list.items():
        idx = np.argmax([v[1] for v in val])
        final[key] = val[idx][1]
        scores[key] = val[idx][0]
    return final, scores


def merge_answers(ids, golds):
    gold_list = {}
    for idx, uid in enumerate(ids):
        gold = golds[idx]
        if not uid in gold_list:
            gold_list[uid] = gold
    return gold_list
