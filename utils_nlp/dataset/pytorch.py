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
