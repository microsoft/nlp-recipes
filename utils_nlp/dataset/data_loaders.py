import random
import dask.dataframe as dd


class DaskCSVLoader:
    """Class for creating and using a loader for large csv
        or other delimited files. The loader uses dask to read
        smaller partitions of a file into memory (one partition at a time),
        before sampling batches from the partitions.
    """

    def __init__(
        self,
        file_path,
        sep=",",
        header="infer",
        block_size=10e6,
        random_seed=None,
    ):
        """Initializes the loader.

        Args:
            file_path (str): Path to delimited file.
            sep (str, optional): Delimiter. Defaults to ",".
            header (str, optional): Number of rows to be used as the header.
                See pandas.read_csv()
                Defaults to "infer".
            block_size (int, optional): Size of partition in bytes.
                See dask.dataframe.read_csv()
                Defaults to 10e6.
            random_seed (int, optional): Random seed. See random.seed().
                Defaults to None.
        """

        self.df = dd.read_csv(
            file_path, sep=sep, header=header, blocksize=block_size
        )
        self.random_seed = random_seed
        random.seed(random_seed)

    def get_random_batches(self, num_batches, batch_size):
        """Creates a random-batch generator.
            Batches returned are pandas dataframes of length=batch_size.
            Note: If the sampled partition has less rows than the
            specified batch_size, then a smaller batch of the same
            size as that partition's number of rows is returned.

        Args:
            num_batches (int): Number of batches to generate.
            batch_size (int]): Batch size.
        """
        for i in range(num_batches):
            rnd_part_idx = random.randint(0, self.df.npartitions - 1)
            sample_part = self.df.partitions[rnd_part_idx].compute()
            if sample_part.shape[0] > batch_size:
                yield sample_part.sample(
                    batch_size, random_state=self.random_seed
                )
            else:
                yield sample_part

    def get_sequential_batches(self, batch_size):
        """Creates a sequential generator.
            Batches returned are pandas dataframes of length=batch_size.
            Note: Final batch might be of smaller size.

        Args:
            batch_size (int): Batch size.
        """
        for i in range(self.df.npartitions):
            part = self.df.partitions[i].compute()
            for j in range(0, part.shape[0], batch_size):
                yield part.iloc[j : j + batch_size, :]
