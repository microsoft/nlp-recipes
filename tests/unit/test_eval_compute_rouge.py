import os
import pytest
from utils_nlp.eval import compute_rouge_perl, compute_rouge_python

ABS_TOL = 0.00001

R1R = 0.71429
R1P = 0.77381
R1F = 0.74176
R2R = 0.44231
R2P = 0.49231
R2F = 0.46504
RLR = 0.67857
RLP = 0.73810
RLF = 0.70605

R1R_hi = 0.53571
R1P_hi = 0.68125
R1F_hi = 0.59804
R2R_hi = 0.28431
R2P_hi = 0.38334
R2F_hi = 0.325
RLR_hi = 0.53571
RLP_hi = 0.68125
RLF_hi = 0.59804


@pytest.fixture()
def rouge_test_data():
    ## First English testing case:
    # Unigrams in candidate: 14
    # Unigrams in reference: 14
    # Unigram overlapping: 10
    # Bigrams in candidate: 13
    # Bigrams in reference: 13
    # Bigram overlapping: 5
    # LCS: 6, 3
    # ROUGE-1 R: 10/14 = 0.71429
    # ROUGE-1 P: 10/14 = 0.71429
    # ROUGE-1 F: 2/(14/10 + 14/10) = 20/28 = 0.71429
    # ROUGE-2 R: 5/13 = 0.38462
    # ROUGE-2 P: 5/13 = 0.38462
    # ROUGE-2 F: 0.38462
    # ROUGE-L R: (6+3)/(9+5) = 0.64286
    # ROUGE-L P: 0.64286
    # ROUGE-L F: 0.64286

    ## Second English testing case:
    # Unigrams in candidate: 6
    # Unigrams in reference: 7
    # Unigram overlapping: 5
    # Bigrams in candidate: 5
    # Bigrams in reference: 6
    # Bigram overlapping: 3
    # LCS: 5
    # ROUGE-1 R: 5/7 = 0.71429
    # ROUGE-1 P: 5/6 = 0.83333
    # ROUGE-1 F: 2/(7/5 + 6/5) = 10/13 = 0.76923
    # ROUGE-2 R: 3/6 = 0.5
    # ROUGE-2 P: 3/5 = 0.6
    # ROUGE-2 F: 2/(6/3 + 5/3) = 6/11 = 0.54545
    # ROUGE-L R: 5/7 = 0.71429
    # ROUGE-L P: 5/6 = 0.83333
    # ROUGE-L F: 2/(7/5 + 6/5) = 10/13 = 0.76923

    summary_candidates = [
        "The stock market is doing very well this year. Hope the same for 2020",
        "The new movie is very popular.",
    ]
    summary_references = [
        "The stock market is doing really well in 2019. Hope 2020 is the same.",
        "The movie is very popular among millennials.",
    ]

    ## First Hindi testing case:
    # Unigrams in candidate: 16
    # Unigrams in reference: 18
    # Unigram overlapping: 9
    # Bigrams in candidate: 15
    # Bigrams in reference: 17
    # Bigram overlapping: 4
    # LCS: 6, 3 (for each reference sentence, the code checks each candidate sentence)
    # ROUGE-1 R: 9/18 = 0.5
    # ROUGE-1 P: 9/16 = 0.5625
    # ROUGE-1 F: 2/(18/9 + 16/9) = 18/34 = 0.52941
    # ROUGE-2 R: 4/17 = 0.23529
    # ROUGE-2 P: 4/15 = 0.26667
    # ROUGE-2 F: 2/(17/4 + 15/4) = 8/32 = 0.25
    # ROUGE-L R: (6+3)/18 = 0.5
    # ROUGE-L P: (6+3)/16 = 0.5625
    # ROUGE-L F: 2/(18/9 + 16/9) = 18/34 = 0.52941

    ## Second Hindi testing case:
    # Unigrams in candidate: 5
    # Unigrams in reference: 7
    # Unigram overlapping: 4
    # Bigrams in candidate: 4
    # Bigrams in reference: 6
    # Bigram overlapping: 2
    # LCS: 4
    # ROUGE-1 R: 4/7 = 0.57143
    # ROUGE-1 P: 4/5 = 0.8
    # ROUGE-1 F: 2/(7/4 + 5/4) = 8/12 = 0.66667
    # ROUGE-2 R: 2/6 = 0.33333
    # ROUGE-2 P: 2/4 = 0.5
    # ROUGE-2 F: 2/(6/2 + 4/2) = 4/10 = 0.4
    # ROUGE-L R: 4/7 = 0.57143
    # ROUGE-L P: 4/5 = 0.8
    # ROUGE-L F: 2/(7/4 + 5/4) = 8/12 = 0.66667

    summary_candidates_hi = [
        "शेयर बाजार इस साल बहुत अच्छा कर रहा है। 2020 के लिए भी यही उम्मीद है।",
        "नई फिल्म बहुत लोकप्रिय है।",
    ]
    summary_references_hi = [
        "शेयर बाजार 2019 में वास्तव में अच्छा कर रहा है। आशा है कि 2020 भी ऐसा ही होगा।",
        "फिल्म सदियों के बीच बहुत लोकप्रिय है।",
    ]

    return {
        "candidates": summary_candidates,
        "references": summary_references,
        "candidates_hi": summary_candidates_hi,
        "references_hi": summary_references_hi,
    }


def test_compute_rouge_perl(rouge_test_data):
    rouge_perl = compute_rouge_perl(
        cand=rouge_test_data["candidates"], ref=rouge_test_data["references"]
    )

    pytest.approx(rouge_perl["rouge_1_recall"], R1R, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_1_precision"], R1P, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_1_f_score"], R1F, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_2_recall"], R2R, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_2_precision"], R2P, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_2_f_score"], R2F, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_l_recall"], RLR, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_l_precision"], RLP, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_l_f_score"], RLF, abs=ABS_TOL)


def test_compute_rouge_python(rouge_test_data):
    rouge_python = compute_rouge_python(
        cand=rouge_test_data["candidates"], ref=rouge_test_data["references"]
    )

    pytest.approx(rouge_python["rouge-1"]["r"], R1R, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-1"]["p"], R1P, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-1"]["f"], R1F, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-2"]["r"], R2R, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-2"]["p"], R2P, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-2"]["f"], R2F, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-l"]["r"], RLR, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-l"]["p"], RLP, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-l"]["f"], RLF, abs=ABS_TOL)


def test_compute_rouge_python_hi(rouge_test_data):
    rouge_python = compute_rouge_python(
        cand=rouge_test_data["candidates_hi"], ref=rouge_test_data["references_hi"], language="hi"
    )

    pytest.approx(rouge_python["rouge-1"]["r"], R1R_hi, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-1"]["p"], R1P_hi, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-1"]["f"], R1F_hi, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-2"]["r"], R2R_hi, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-2"]["p"], R2P_hi, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-2"]["f"], R2F_hi, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-l"]["r"], RLR_hi, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-l"]["p"], RLP_hi, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-l"]["f"], RLF_hi, abs=ABS_TOL)


def test_compute_rouge_perl_file(rouge_test_data, tmp):
    tmp_cand_file = os.path.join(tmp, "cand.txt")
    tmp_ref_file = os.path.join(tmp, "ref.txt")

    with open(tmp_cand_file, "w") as f:
        for s in rouge_test_data["candidates"]:
            f.write(s + "\n")
    with open(tmp_ref_file, "w") as f:
        for s in rouge_test_data["references"]:
            f.write(s + "\n")

    rouge_perl = compute_rouge_perl(cand=tmp_cand_file, ref=tmp_ref_file, is_input_files=True)

    pytest.approx(rouge_perl["rouge_1_recall"], R1R, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_1_precision"], R1P, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_1_f_score"], R1F, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_2_recall"], R2R, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_2_precision"], R2P, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_2_f_score"], R2F, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_l_recall"], RLR, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_l_precision"], RLP, abs=ABS_TOL)
    pytest.approx(rouge_perl["rouge_l_f_score"], RLF, abs=ABS_TOL)


def test_compute_rouge_python_file(rouge_test_data, tmp):
    tmp_cand_file = os.path.join(tmp, "cand.txt")
    tmp_ref_file = os.path.join(tmp, "ref.txt")

    with open(tmp_cand_file, "w") as f:
        for s in rouge_test_data["candidates"]:
            f.write(s + "\n")
    with open(tmp_ref_file, "w") as f:
        for s in rouge_test_data["references"]:
            f.write(s + "\n")

    rouge_python = compute_rouge_python(cand=tmp_cand_file, ref=tmp_ref_file, is_input_files=True)

    pytest.approx(rouge_python["rouge-1"]["r"], R1R, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-1"]["p"], R1P, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-1"]["f"], R1F, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-2"]["r"], R2R, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-2"]["p"], R2P, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-2"]["f"], R2F, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-l"]["r"], RLR, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-l"]["p"], RLP, abs=ABS_TOL)
    pytest.approx(rouge_python["rouge-l"]["f"], RLF, abs=ABS_TOL)
