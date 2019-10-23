""" Official evaluation script for SQuAD version 2.0.
    Modified by XLNet authors to update `find_best_threshold` scripts for SQuAD V2.0
"""

import collections
import json
import re
import string


def get_raw_scores(qa_ids, actuals, preds):
    """
        Computes exact match and F1 scores without applying any unanswerable probability threshold.

        Args:
            qa_ids (list): Unique ids corresponding to the answers in `actuals`.
            actuals (list): List of ground truth answers.
            preds (dict): Dictionary with qa_id as keys and predicted answers as values.

        Returns:
            tuple: (exact_match, f1)

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

    # Helper functions end

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
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        exact_scores[qid] = max(_compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(_compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans, unanswerable_exists=False):
    """
    Find the best threshold to determine a question is impossible to answer.

    Args:
        preds (dict): Dictionary with qa_id as keys and predicted answers as values.
        scores (dict): Dictionary with qa_id as keys and raw evaluation scores (exact_match or
            f1) as values.
        na_probs (dict): Dictionary with qa_id as keys and unanswerable probabilities as values.
        qid_to_has_ans (dict): Dictionary with qa_id as keys boolean values indicating if the
            question has answer as values.
        unanswerable_exists (bool, optional): Whether there is unanswerable questions in the data.
            Defaults to False.

    Returns:
        tuple: score after applying best threshold, best threshold, (score for answerable
            questions after applying best threshold, if unanswerable_exists=True)
    """
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    # If na_prob > threshold, the question is considered as unanswerable by the prediction.
    # Initially, the threshold is 0. All questions are considered as unanswerable by the
    # predictions. So cur_score is the number of actual unanswerable questions (i.e. correctly
    # predicted as unanswerable in the data.
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0

    # Sorted in ascending order
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        # When using the cur_na_prob as threshold, all predictions with na_prob > na_prob_cur are
        # considered as unanswerable. Current question is considered answerable.
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            # Current question has ground truth answer, the prediction is correct. The raw score
            # is added to cur_score
            diff = scores[qid]
        else:
            # Current question doesn't have ground truth answer.
            if preds[qid]:
                # Prediction is not empty, incorrect. cur_score -= 1
                diff = -1
            else:
                # Prediction is empty, correct, the original score 1 from num_no_ans is preserved.
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            # When cur_score > best_score, the threshold can increase so that more questions are
            # considered as answerable and fewer questions are considered as unanswerable.
            # Imagine a PDF with two humps with some overlapping, the x axis is the na_prob. The
            # hump on the left is answerable questions and the hump on the right is unanswerable
            # questions.
            # At some point, the number of actual answerable questions decreases, and we got more
            # penalty from considering unanswerable questions as answerable than the score added
            # from actual answerable questions, we will not change the threshold anymore and the
            # optimal threshold is found.
            best_score = cur_score
            best_thresh = na_probs[qid]

    if not unanswerable_exists:
        return 100.0 * best_score / len(scores), best_thresh
    else:
        has_ans_score, has_ans_cnt = 0, 0
        for qid in qid_list:
            if not qid_to_has_ans[qid]:
                continue
            has_ans_cnt += 1

            if qid not in scores:
                continue
            has_ans_score += scores[qid]

        return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def find_all_best_thresh(
    main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans, unanswerable_exists=False
):
    """
    Update raw evaluation scores by finding the best threshold to determine a question is
    impossible to answer.

    Args:
        main_eval (dict): Dictionary with raw evaluation scores without apply any threshold.
        preds (dict): Dictionary with qa_id as keys and predicted answers as values.
        exact_raw (dict): Dictionary with qa_id as keys and raw exact_match scores as values.
        f1_raw (dict): Dictionary with qa_id as keys and raw f1 scores as values.
        na_probs (dict): Dictionary with qa_id as keys and unanswerable probabilities as values.
        qid_to_has_ans (dict): Dictionary with qa_id as keys boolean values indicating if the
            question has answer as values.
        unanswerable_exists (bool, optional): Whether there is unanswerable questions in the data.
            Defaults to False.

    Returns:
        dict: Updated `main_eval` with scores after applying best threshold and best threshold
            for each score.
    """
    all_exact = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans, unanswerable_exists)
    all_f1 = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans, unanswerable_exists)
    main_eval["best_exact"] = all_exact[0]
    main_eval["best_exact_thresh"] = all_exact[1]
    main_eval["best_f1"] = all_f1[0]
    main_eval["best_f1_thresh"] = all_f1[1]

    if unanswerable_exists:
        main_eval["has_ans_exact"] = all_exact[2]
        main_eval["has_ans_f1"] = all_f1[2]


def evaluate_qa(
    actual_dataset, preds, na_probs=None, na_prob_thresh=0, unanswerable_exists=False, out_file=None
):
    """
    Evaluate question answering prediction results against ground truth answers.

    Args:
        Evaluates question answering model performance.

        Args:
            actual_dataset (:class:`utils_nlp.dataset.pytorch.QADataset`): Input question answering
                dataset with ground truth answers.
            preds (dict): The key of the dictionary is the qa_id in the original
                :class:`utils_nlp.dataset.pytorch.QADataset`. The values of the dictionary are
                the predicted answer texts in string type.
            na_probs (dict, optional): Dictionary of qa_id and unanswerable probability pairs.
                If None, unanswerable probabilities are all set to zero. Defaults to None.
            na_prob_thresh (float, optional): Probability threshold to predict a question to be
                unanswerable. For an unanswerable question, if `na_probs` > `na_prob_thresh`,
                the prediction is considered as correct. Otherwise, the prediction is considered as
                incorrect. Defaults to 0.
            out_file (str, optional): Path of the file to save the evaluation results to.
                Defaults to None.

        Returns:
            dict: A dictionary with exact_match and f1 values.
    """

    # Helper functions
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
        na_probs_available = False
        na_probs = {k: 0.0 for k in preds}
    else:
        na_probs_available = True

    qa_ids = [item.qa_id for item in actual_dataset]
    actuals = [item.answer_text for item in actual_dataset]

    qid_to_has_ans = {qa_id: bool(ans) for (qa_id, ans) in zip(qa_ids, actuals)}
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(qa_ids, actuals, preds)
    exact_thresh = _apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, na_prob_thresh)
    f1_thresh = _apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, na_prob_thresh)
    out_eval = _make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = _make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
        _merge_eval(out_eval, has_ans_eval, "HasAns")
    if no_ans_qids:
        no_ans_eval = _make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
        _merge_eval(out_eval, no_ans_eval, "NoAns")

    if na_probs_available:
        find_all_best_thresh(
            out_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans, unanswerable_exists
        )

    if out_file:
        with open(out_file, "w") as f:
            json.dump(out_eval, f)
    else:
        print(json.dumps(out_eval, indent=2))
    return out_eval
