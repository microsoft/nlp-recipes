import os
import shutil
import time
import tempfile

from pyrouge import Rouge155
from rouge import Rouge


def compute_rouge_perl(cand, ref, input_files=False):

    temp_dir = tempfile.mkdtemp()

    if input_files:
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
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w", encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w", encoding="utf-8") as f:
                f.write(references[i])
        r = Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = "ref.#ID#.txt"
        r.system_filename_pattern = r"cand.(\d+).txt"
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def compute_rouge_python(cand, ref, input_files=False):
    if input_files:
        candidates = [line.strip() for line in open(cand, encoding="utf-8")]
        references = [line.strip() for line in open(ref, encoding="utf-8")]
    else:
        candidates = cand
        references = ref

    print("Number of candidates: {}".format(len(candidates)))
    print("Number of references: {}".format(len(references)))
    assert len(candidates) == len(references)

    evaluator = Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True)

    scores = evaluator.get_scores(candidates, [[it] for it in references])

    return scores
