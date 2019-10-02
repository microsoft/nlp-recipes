import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from utils_nlp.dataset import Split
from utils_nlp.dataset.multinli import load_pandas_df



temp_dir = "/media/bleik2/temp"
sample_size = 20000
train_set_proportion = 0.75

df = load_pandas_df(temp_dir, "train")
df = df[df["gold_label"] == "neutral"]
df = df.sample(sample_size)
df_train, df_test = train_test_split(df, train_size=train_set_proportion)
text_train = df_train["sentence1"]
text_test = df_test["sentence1"]
label_encoder = LabelEncoder()
labels_train = label_encoder.fit_transform(df_train["genre"])
labels_test = label_encoder.transform(df_test["genre"])

#############s
tf_idf = TfidfVectorizer(
    use_idf=True,
    ngram_range=(1, 2),
    max_features=None,
    max_df=5000,
    min_df=5,
    lowercase=True,
    stop_words="english",
).fit(text_train)

classifier = svm.LinearSVC()
pipeline = Pipeline([("tf_idf", tf_idf), ("classifier", classifier)])
trained = pipeline.fit(text_train, labels_train)

preds = trained.predict(text_test)
print("accuracy: {}".format(accuracy_score(labels_test, preds)))
print(classification_report(labels_test, preds, labels=label_encoder.classes_))
#############

from utils_nlp.models.transformers.sequence_classification import SequenceClassifier, Processor

# list supported pre-trained models
pd.DataFrame(SequenceClassifier.list_supported_models())

model_name = "roberta-base"
max_len = 128
num_labels = len(label_encoder.classes_)
# preprocess
processor = Processor(model_name=model_name, cache_dir=temp_dir)
ds_train = processor.preprocess(text_train, labels_train, max_len=max_len)
ds_test = processor.preprocess(text_test, None, max_len=max_len)
# fine_tune
classifier = SequenceClassifier(model_name=model_name, num_labels=num_labels, cache_dir=temp_dir)
classifier.fit(ds_train, device="cuda", num_epochs=1, batch_size=16, num_gpus=2)
# predict
preds = classifier.predict(ds_test, device="cuda", batch_size=32, num_gpus=2)
# eval
print("accuracy: {}".format(accuracy_score(labels_test, preds)))
print(classification_report(labels_test, preds, target_names=label_encoder.classes_))
temp['macro avg']["f1-score"]

temp = classification_report([1,2,3,1,2,3,1,2,3],[1,2,2,1,2,3,1,2,1], output_dict=True)

temp['macro avg']["f1-score"]