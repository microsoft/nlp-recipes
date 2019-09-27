from torch.utils.data import Dataset


class SCDataSet(Dataset):
    def __init__(self, df, text_col, label_col):
        self.df = df

        if isinstance(text_col, int):
            self.text_col = text_col
        elif isinstance(text_col, str):
            self.text_col = df.columns.index(text_col)
        else:
            raise TypeError("text_col must be of type int or str")

        if isinstance(label_col, int):
            self.label_col = label_col
        elif isinstance(label_col, str):
            self.label_col = df.columns.index(label_col)
        else:
            raise TypeError("label_col must be of type int or str")

    def __getitem__(self, idx):
        return (self.df.iloc[idx, self.text_col], self.df.iloc[idx, self.label_col])

    def __len__(self):
        return self.df.shape[0]


QAInput = collections.namedtuple(
    "QAInput",
    ["doc_text", "question_text", "qa_id", "is_impossible", "answer_start", "answer_text"],
)


class QADataset(Dataset):
    def __init__(
        self,
        df,
        doc_text_col,
        question_text_col,
        qa_id_col,
        is_impossible_col=None,
        answer_start_col=None,
        answer_text_col=None,
    ):

        self.df = df.copy()
        self.doc_text_col = doc_text_col
        self.question_text_col = question_text_col

        ## TODO: can this be made optional???
        ## Yes, if we make evaluate_qa takes QADataset.
        self.qa_id_col = qa_id_col

        if is_impossible_col is None:
            self.is_impossible_col = "is_impossible"
            df[self.is_impossible_col] = False
        else:
            self.is_impossible_col = is_impossible_col

        if answer_start_col is not None and answer_text_col is not None:
            self.actual_answer_available = True
        else:
            self.actual_answer_available = False
        self.answer_start_col = answer_start_col
        self.answer_text_col = answer_text_col

    def __getitem__(self, idx):
        current_item = self.df.iloc[idx, ]
        if self.actual_answer_available:
            return QAInput(
                doc_text=current_item[self.doc_text_col],
                question_text=current_item[self.question_text_col],
                qa_id=current_item[self.qa_id_col],
                is_impossible=current_item[self.is_impossible_col],
                answer_start=current_item[self.answer_start_col],
                answer_text=current_item[self.answer_text_col],
            )
        else:
            return QAInput(
                doc_text=current_item[self.doc_text_col],
                question_text=current_item[self.question_text_col],
                qa_id=current_item[self.qa_id_col],
                is_impossible=current_item[self.is_impossible_col],
                answer_start=-1,
                answer_text="",
            )

    def __len__(self):
        return self.df.shape[0]
