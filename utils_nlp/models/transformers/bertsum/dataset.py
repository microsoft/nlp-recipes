import itertools
import torch
from torch.utils.data import (
    Dataset,
    IterableDataset,
)


def get_dataset(file):
    yield torch.load(file)


class ExtSumProcessedIterableDataset(IterableDataset):
    """Iterable dataset for extractive summarization preprocessed data
    """

    def __init__(self, file_list, is_shuffle=False):
        """ Initiation function for iterable dataset for extractive summarization
            preprocessed data.

        Args:
            file_list (list of strings): List of files that the dataset is loaded from.
            is_shuffle (bool, optional): A boolean value specifies whether the list of
                files is shuffled when the dataset is loaded. Defaults to False.
        """

        self.file_list = file_list
        self.is_shuffle = is_shuffle

    def get_stream(self):
        """ get a stream of cycled data from the dataset"""

        if self.is_shuffle:
            return itertools.chain.from_iterable(
                map(get_dataset, itertools.cycle(self.file_list))
            )
        else:
            return itertools.chain.from_iterable(
                map(get_dataset, itertools.cycle(random.shuffle(self.file_list)))
            )

    def __iter__(self):
        return self.get_stream()


class ExtSumProcessedDataset(Dataset):
    """Dataset for extractive summarization preprocessed data
    """

    def __init__(self, file_list, is_shuffle=False):
        """ Initiation function for dataset for extractive summarization preprocessed data.

        Args:
            file_list (list of strings): List of files that the dataset is loaded from.
            is_shuffle (bool, optional): A boolean value specifies whether the list of
                files is shuffled when the dataset is loaded. Defaults to False.
        """

        self.file_list = sorted(file_list)
        if is_shuffle:
            random.shuffle(self.file_list)
        self.data = []
        for f in self.file_list:
            self.data.extend(torch.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
