import os
import pickle
import shutil
import sys

import torch
from utils_nlp.models.transformers.sequence_classification import SequenceClassifier

print("CUDA is{} available".format("" if torch.cuda.is_available() else " not"))

input_dir = sys.argv[1]
output_dir = sys.argv[2]
model_name = sys.argv[3]
num_gpus = int(sys.argv[4])
cache_dir = "."
num_labels = 5
batch_size = 16

if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)

# load dataset
ds = pickle.load(open(os.path.join(input_dir, model_name + "_ds"), "rb"))
# fine-tune
classifier = SequenceClassifier(model_name=model_name, num_labels=num_labels, cache_dir=cache_dir)
classifier.fit(ds, batch_size=batch_size, num_gpus=num_gpus, verbose=False)
# write classifier
pickle.dump(classifier, open(os.path.join(output_dir, model_name + "_clf"), "wb"))
# write label encoder
shutil.move(
    os.path.join(input_dir, model_name + "_le"), os.path.join(output_dir, model_name + "_le")
)
