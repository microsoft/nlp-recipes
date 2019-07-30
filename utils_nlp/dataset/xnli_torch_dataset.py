import numpy as np
import torch
from utils_nlp.models.bert.common import Language, Tokenizer
from torch.utils import data
from utils_nlp.dataset.xnli import load_pandas_df
from sklearn.preprocessing import LabelEncoder

MAX_SEQ_LENGTH = 128
TEXT_COL = "text"
LABEL_COL = "label"
#DATA_USED_PERCENT = 0.0025
TRAIN_FILE_SPLIT = "train"
TEST_FILE_SPLIT = "test"
VALIDATION_FILE_SPLIT = "dev"
CACHE_DIR = "./"
LANGUAGE_ENGLISH = "en"
TO_LOWER_CASE = False
TOK_ENGLISH = Language.ENGLISH


class XnliDataset(data.Dataset):
    def __init__(
        self,
        file_split=TRAIN_FILE_SPLIT,
        cache_dir=CACHE_DIR,
        language=LANGUAGE_ENGLISH,
        to_lowercase=TO_LOWER_CASE,
        tok_language=TOK_ENGLISH,
    ):

        self.file_split = file_split
        self.cache_dir = cache_dir
        self.language = language
        self.to_lowercase = to_lowercase
        self.tok_language = tok_language

        df = load_pandas_df(
            local_cache_path=cache_dir,
            file_split=file_split,
            language=language,
        )

        # if file_split == TRAIN_FILE_SPLIT:
        #     data_used_count = round(DATA_USED_PERCENT * df.shape[0])
        #     df = df.loc[:data_used_count]

        self.df = df

        print("===================df length===================", len(self.df))

        print("Create a tokenizer...")
        tokenizer = Tokenizer(
            language=tok_language, to_lower=to_lowercase, cache_dir=cache_dir
        )
        tokens = tokenizer.tokenize(df[TEXT_COL])

        print("Tokenize and preprocess text...")
        # tokenize
        token_ids, input_mask, token_type_ids = tokenizer.preprocess_classification_tokens(
            tokens, max_len=MAX_SEQ_LENGTH
        )

        # preprocess
        self.token_ids = token_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids

        if file_split == TRAIN_FILE_SPLIT:
            label_encoder = LabelEncoder()
            train_labels = label_encoder.fit_transform(df[LABEL_COL])
            self.label_encoder = label_encoder
            self.labels = np.array(train_labels)

        if file_split == TEST_FILE_SPLIT:
            # use the label_encoder passed when you create the test dataset
            self.labels = df[LABEL_COL]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        token_ids = self.token_ids[index]
        input_mask = self.input_mask[index]
        token_type_ids = self.token_type_ids[index]
        labels = self.labels[index]

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "input_mask": torch.tensor(input_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": labels,
        }
