import pandas as pd
import numpy as np
import random
import pickle
import urllib3
import tarfile
import io


URL = "https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz"


def download(dir_path):
    con = urllib3.PoolManager()
    resp = con.request("GET", URL)
    tar = tarfile.open(fileobj=io.BytesIO(resp.data))
    tar.extractall(path=dir_path)
    tar.close()


def read_data(data_file, nrows=None):
    return pd.read_csv(data_file, header=None, nrows=nrows)


def clean_data(df):
    df.fillna("", inplace=True)
    text = df.iloc[:, 1] + " " + df.iloc[:, 2] + " " + df.iloc[:, 3]
    text = text.str.replace(r"[^A-Za-z ]", "").str.lower()
    text = text.str.replace(r"\\s+", " ")
    text = text.astype(str)
    return text


def get_labels(df):
    return list(df[0] - 1)


def get_batch_rnd(X, input_mask, y, n, batch_size):
    i = int(random.random() * n)
    X_b = X[i : i + batch_size]
    y_b = y[i : i + batch_size]
    m_b = input_mask[i : i + batch_size]
    return X_b, m_b, y_b


def get_batch_by_idx(X, input_mask, y, i, batch_size):
    X_b = X[i : i + batch_size]
    y_b = y[i : i + batch_size]
    m_b = input_mask[i : i + batch_size]    
    return X_b, m_b, y_b

