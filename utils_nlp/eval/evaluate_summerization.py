import os
from random import random, seed
from others.utils import test_rouge

def get_rouge(predictions, targets, temp_dir):
    def _write_list_to_file(list_items, filename):
        with open(filename, 'w') as filehandle:
            #for cnt, line in enumerate(filehandle):
            for item in list_items:
                filehandle.write('%s\n' % item)
    seed(42)
    random_number = random()
    candidate_path = os.path.join(temp_dir, "candidate"+str(random_number))
    gold_path = os.path.join(temp_dir, "gold"+str(random_number))
    _write_list_to_file(predictions, candidate_path)
    _write_list_to_file(targets, gold_path)
    rouge = test_rouge(temp_dir, candidate_path, gold_path)
    return rouge

