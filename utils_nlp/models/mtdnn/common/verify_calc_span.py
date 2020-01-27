from pytorch_pretrained_bert import BertTokenizer
from data_utils.task_def import EncoderModelType
from experiments.squad.squad_utils import calc_tokenized_span_range, parse_squad_label

model = "bert-base-uncased"
do_lower_case = True
tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=do_lower_case)

for no, line in enumerate(open(r"data\canonical_data\squad_v2_train.tsv", encoding="utf-8")):
    if no % 1000 == 0:
        print(no)
    uid, label, context, question = line.strip().split("\t")
    answer_start, answer_end, answer, is_impossible = parse_squad_label(label)
    calc_tokenized_span_range(context, question, answer, answer_start, answer_end, tokenizer, EncoderModelType.BERT,
                              verbose=True)
