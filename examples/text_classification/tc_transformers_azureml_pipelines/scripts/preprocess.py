from utils_nlp.models.transformers.sequence_classification import SequenceClassifier, Processor
import sys
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder


input_dir = sys.argv[1]
input_data_file = sys.argv[2]
output_dir = sys.argv[3]
text_col = sys.argv[4]
label_col = sys.argv[5]
model_name = sys.argv[6]
max_len = 150
cache_dir = "."


if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)

# read data
df = pd.read_csv(os.path.join(input_dir, input_data_file), quoting=1)

# encode labels

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df[label_col])

# preprocess
processor = Processor(model_name=model_name, cache_dir=cache_dir)
ds = processor.preprocess(df[text_col], labels, max_len=max_len)

# write preprocessed dataset
output_data_file = model_name + "_ds"
pickle.dump(ds, open(os.path.join(output_dir, output_data_file), "wb"))

# write label encoder
label_encoder_file = model_name + "_le"
pickle.dump(label_encoder, open(os.path.join(output_dir, label_encoder_file), "wb"))
