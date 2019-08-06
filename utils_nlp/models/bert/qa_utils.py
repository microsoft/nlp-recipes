# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script reuses some code from
# https://github.com/huggingface/pytorch-transformers/blob/master/examples/utils_squad.py
# https://github.com/huggingface/pytorch-transformers/blob/master/examples/utils_squad_evaluate.py

import collections
import json
import math
import logging
import re
import string

logger = logging.getLogger(__name__)

from pytorch_transformers.tokenization_bert import BasicTokenizer

QAExample = collections.namedtuple(
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

QAFeatures = collections.namedtuple(
    "QAFeatures",
    [
        "unique_id",
        "example_index",
        "tokens",
        "token_to_orig_map",
        "token_is_max_context",
        "input_ids",
        "input_mask",
        "segment_ids",
        "start_position",
        "end_position",
        "paragraph_len",
    ],
)

QAResult = collections.namedtuple(
    "QAResult", ["unique_id", "start_logits", "end_logits"]
)


def postprocess_answers(
    all_results,
    all_examples,
    all_features,
    do_lower_case,
    n_best_size=20,
    max_answer_length=30,
    output_prediction_file="./qa_predictions.json",
    output_nbest_file="./nbest_predictions.json",
    unanswerable_exists=False,
    output_null_log_odds_file=None,
    null_score_diff_threshold=0.0,
    verbose_logging=False,
):

    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    # Map unique features to the original doc-question-answer triplet
    # Each doc-question-answer triplet can have multiple features because the doc
    # could be split into multiple spans
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        [
            "feature_index",
            "start_index",
            "end_index",
            "start_logit",
            "end_logit",
        ],
    )

    all_predictions = collections.OrderedDict()
    all_probs = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        # get all the features belong to the same example,
        # i.e. paragaraph/question pair.
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = (
            0
        )  # the start logit at the slice with min null score
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
                feature_null_score = (
                    result.start_logits[0] + result.end_logits[0]
                )
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
                    if not feature.token_is_max_context.get(
                        start_index, False
                    ):
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
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
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
                tok_tokens = feature.tokens[
                    pred.start_index : (pred.end_index + 1)
                ]
                ## Why can't we use span tokens directly???
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[
                    orig_doc_start : (orig_doc_end + 1)
                ]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, do_lower_case, verbose_logging
                )
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                )
            )
        # if we didn't include the empty option in the n-best, include it
        if unanswerable_exists:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                    )
                )

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(
                    0,
                    _NbestPrediction(
                        text="empty", start_logit=0.0, end_logit=0.0
                    ),
                )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0)
            )

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
                score_null
                - best_non_null_entry.start_logit
                - (best_non_null_entry.end_logit)
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
        if not output_null_log_odds_file:
            output_null_log_odds_file = "./null_odds.json"
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, all_probs, nbest_json


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
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text)
            )
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
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True
    )

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


## Evaluation utilities
def evaluate_qa(
    qa_ids, actuals, preds, na_probs=None, na_prob_thresh=0, out_file=None
):
    # with open(data_file) as f:
    #     dataset_json = json.load(f)
    #     dataset = dataset_json['data']
    # with open(pred_file) as f:
    #     preds = json.load(f)
    # if na_prob_file:
    #     with open(na_prob_file) as f:
    #         na_probs = json.load(f)
    # else:
    #     na_probs = {k: 0.0 for k in preds}

    if na_probs is None:
        na_probs = {k: 0.0 for k in preds}

    qid_to_has_ans = {
        qa_id: bool(ans) for (qa_id, ans) in zip(qa_ids, actuals)
    }
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(qa_ids, actuals, preds)
    exact_thresh = apply_no_ans_threshold(
        exact_raw, na_probs, qid_to_has_ans, na_prob_thresh
    )
    f1_thresh = apply_no_ans_threshold(
        f1_raw, na_probs, qid_to_has_ans, na_prob_thresh
    )
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(
            exact_thresh, f1_thresh, qid_list=has_ans_qids
        )
        merge_eval(out_eval, has_ans_eval, "HasAns")
    if no_ans_qids:
        no_ans_eval = make_eval_dict(
            exact_thresh, f1_thresh, qid_list=no_ans_qids
        )
        merge_eval(out_eval, no_ans_eval, "NoAns")

    if out_file:
        with open(out_file, "w") as f:
            json.dump(out_eval, f)
    else:
        print(json.dumps(out_eval, indent=2))
    return out_eval


def normalize_answer(s):
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


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
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


def get_raw_scores(qa_ids, actuals, preds):
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
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
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
                (
                    "exact",
                    100.0 * sum(exact_scores[k] for k in qid_list) / total,
                ),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]
