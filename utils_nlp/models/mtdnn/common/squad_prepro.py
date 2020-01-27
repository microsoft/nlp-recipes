import os
import argparse
from sys import path
import json
path.append(os.getcwd())
from data_utils.log_wrapper import create_logger
from experiments.common_utils import dump_rows
from data_utils import DataFormat

logger = create_logger(__name__, to_disk=True, log_file='squad_prepro.log')

def normalize_qa_field(s: str, replacement_list):
    for replacement in replacement_list:
        s = s.replace(replacement, " " * len(replacement))  # ensure answer_start and answer_end still valid
    return s

#END = 'EOSEOS'
def load_data(path, is_train=True, v2_on=False):
    rows = []
    with open(path, encoding="utf8") as f:
        data = json.load(f)['data']
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if v2_on:
                context = '{} {}'.format(context, END)
            for qa in paragraph['qas']:
                uid, question = qa['id'], qa['question']
                answers = qa.get('answers', [])
                # used for v2.0
                is_impossible = qa.get('is_impossible', False)
                label = 1 if is_impossible else 0
                if (v2_on and label < 1 and len(answers) < 1) or ((not v2_on) and len(answers) < 1):
                    # detect inconsistent data
                    # * for v2, the row is possible but has no answer
                    # * for v1, all questions should have answer
                    continue
                if len(answers) > 0:
                    answer = answers[0]['text']
                    answer_start = answers[0]['answer_start']
                    answer_end = answer_start + len(answer)
                else:
                    # for questions without answers, give a fake answer
                    #answer = END
                    #answer_start = len(context) - len(END)
                    #answer_end = len(context)
                    answer = ''
                    answer_start = -1
                    answer_end = -1
                answer = normalize_qa_field(answer, ["\n", "\t", ":::"])
                context = normalize_qa_field(context, ["\n", "\t"])
                question = normalize_qa_field(question, ["\n", "\t"])
                sample = {'uid': uid, 'premise': context, 'hypothesis': question,
                          'label': "%s:::%s:::%s:::%s" % (answer_start, answer_end, label, answer)}
                rows.append(sample)
    return rows

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing SQUAD data.')
    parser.add_argument('--root_dir', type=str, default='data')
    args = parser.parse_args()
    return args

def main(args):
    root = args.root_dir
    assert os.path.exists(root)

    squad_train_path = os.path.join(root, 'squad/train.json')
    squad_dev_path = os.path.join(root, 'squad/dev.json')
    squad_v2_train_path = os.path.join(root, 'squad_v2/train.json')
    squad_v2_dev_path = os.path.join(root, 'squad_v2/dev.json')

    squad_train_data = load_data(squad_train_path)
    squad_dev_data = load_data(squad_dev_path, is_train=False)
    logger.info('Loaded {} squad train samples'.format(len(squad_train_data)))
    logger.info('Loaded {} squad dev samples'.format(len(squad_dev_data)))

    squad_v2_train_data = load_data(squad_v2_train_path, v2_on=True)
    squad_v2_dev_data = load_data(squad_v2_dev_path, is_train=False, v2_on=True)
    logger.info('Loaded {} squad_v2 train samples'.format(len(squad_v2_train_data)))
    logger.info('Loaded {} squad_v2 dev samples'.format(len(squad_v2_dev_data)))

    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    squad_train_fout = os.path.join(canonical_data_root, 'squad_train.tsv')
    squad_dev_fout = os.path.join(canonical_data_root, 'squad_dev.tsv')
    dump_rows(squad_train_data, squad_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(squad_dev_data, squad_dev_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info('done with squad')

    squad_v2_train_fout = os.path.join(canonical_data_root, 'squad-v2_train.tsv')
    squad_v2_dev_fout = os.path.join(canonical_data_root, 'squad-v2_dev.tsv')
    dump_rows(squad_v2_train_data, squad_v2_train_fout, DataFormat.PremiseAndOneHypothesis)
    dump_rows(squad_v2_dev_data, squad_v2_dev_fout, DataFormat.PremiseAndOneHypothesis)
    logger.info('done with squad_v2')



if __name__ == '__main__':
    args = parse_args()
    main(args)
