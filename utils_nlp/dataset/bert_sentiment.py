from utils_nlp.dataset.url_utils import maybe_download
import tarfile
from tqdm import tqdm_notebook as tqdm
import re
import os
import pandas as pd

# Load the data from a directory

# Download the dataset and load into pandas dataframe
def download_or_find(url, directory=".", filename="aclImdb.tar.gz"):
    """
    Maybe download the data and put it into the given directory with given filename.
    Skip the downloading if file already existed.
    
    Load the data into pandas Dataframe
    Args:
        url (string): The URL of the dataset
        directory (string): Where to look for or store the dataset, default to current directory
        filename (string): What filename to use for retrieve or store the dataset
    
    Return:
        file_path (string): The file_path of the downloaded (or currently exists)
    """
    print("=====> Begin downloading")
    file_path = maybe_download(url, filename, directory)
    print("=====> Done downloading")
    
    data_path = os.path.join(os.getcwd(), directory, "aclImdb")
    
    # Extract the data to the data folder
    if not os.path.exists(data_path):
        tar = tarfile.open(file_path)
        tar.extractall(directory)
        tar.close()
    
    # Return the path of dataset when done 
    print("=====> Finish extracting")
    return data_path
    

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    """
    Method to go through all the files in the directory, get its content and put it into the appropriate train/test
    """
    
    # Create a new dictionary to store initial value for dataframe
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    
    # Loop through all the subdirectories
    for file_path in tqdm(os.listdir(directory)):
        # Open each file
        with open(os.path.join(directory, file_path), "r", encoding="utf8") as f:
            # Each file in the directory will be a text file (.txt) containing a review
            data["sentence"].append(f.read())
            # The name of the file has 2 parts, the index of the file and the sentiment value of that review
            # We only interested in the sentiment value, so only group(1) in the Match object
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    print("===> Directory: {}".format(directory))
    # Load the positive and negative data to pandas Dataframe
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    
    # Denoted positive to be 1 and negative be 0 for classification label
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    
    URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    
    dataset_path = download_or_find(URL, directory="data")

    print("=============> Complete downloading")
    print("**** Dataset path: {}".format(dataset_path))
  
    train_df = load_dataset(os.path.join(dataset_path, "train"))
    
    print("===> Complete train df")

    test_df = load_dataset(os.path.join(dataset_path, "test"))
    print("===> Complete test df")
  
    return train_df, test_df
