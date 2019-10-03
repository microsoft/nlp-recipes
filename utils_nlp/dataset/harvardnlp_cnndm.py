import nltk
nltk.download('punkt')

from nltk import tokenize
import torch
import sys
sys.path.insert(0, '../src')
from others.utils import clean
from multiprocess import Pool

import regex as re
def preprocess(param):
    sentences, preprocess_pipeline, word_tokenize = param
    for function in preprocess_pipeline:
        sentences = function(sentences)
    return [word_tokenize(sentence) for sentence in sentences]


def harvardnlp_cnndm_preprocess(n_cpus, source_file, target_file):
    def _remove_ttags(line):
        line = re.sub(r'<t>', '', line)
        line = re.sub(r'</t>', '', line)
        return line

    def _cnndm_target_sentence_tokenization(line):

        return line.split("</t>")
    src_list = []
    with open(source_file, 'r') as fd:
        for line in  fd:  
            src_list.append((line, [tokenize.sent_tokenize], nltk.word_tokenize))
    pool = Pool(n_cpus)
    tokenized_src_data =  pool.map(preprocess, src_list, int(len(src_list)/n_cpus))
    pool.close()
    pool.join()
    
    tgt_list = []
    with open(target_file, 'r') as fd:
        for line in  fd:  
            tgt_list.append((line, [clean, _remove_ttags, _cnndm_target_sentence_tokenization], nltk.word_tokenize))

    pool = Pool(n_cpus)
    tokenized_tgt_data =  pool.map(preprocess, tgt_list, int(len(tgt_list)/n_cpus))
    pool.close()
    pool.join()

    jobs=[]
    for (src, summary) in zip(tokenized_src_data, tokenized_tgt_data):
        jobs.append({'src': src, "tgt": summary})
    
    return jobs
