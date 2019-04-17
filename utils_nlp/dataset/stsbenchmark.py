import os
import tarfile
import pandas as pd
import azureml.dataprep as dp

from utils_nlp.dataset.url_utils import maybe_download

"""
Download and extract data from http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz 
"""
def download_sts(dirpath):
	sts_url = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
	filepath = maybe_download(sts_url, work_directory = dirpath)
	extracted_path = extract_sts(filepath, target_dirpath = dirpath, tmode = "r:gz")
	print("Data downloaded to {}".format(extracted_path))
	return extracted_path

"""
Extract data from the sts tar.gz archive
"""
def extract_sts(tarpath, target_dirpath = ".", tmode = "r"):
	with tarfile.open(tarpath, mode = tmode) as t:
		t.extractall(target_dirpath)
		extracted = t.getnames()[0]
	os.remove(tarpath)
	return os.path.join(target_dirpath, extracted)

"""
Drop columns containing irrelevant metadata and save as new csv files in the target_dir
"""
def clean_sts(filenames, src_dir, target_dir):
	if not os.path.exists(target_dir):
		os.makedirs(target_dir)
	filepaths = [os.path.join(src_dir, f) for f in filenames]
	for i,fp in enumerate(filepaths):
		dat = dp.auto_read_file(path=fp)
		s = dat.keep_columns(['Column5', 'Column6', 'Column7']).rename_columns({'Column5': 'score', 'Column6': 'sentence1', 'Column7': 'sentence2'})
		sdf = s.to_pandas_dataframe().to_csv(os.path.join(target_dir, filenames[i]), sep='\t')

class STSBenchmark():
	def __init__(self, which_split, base_data_path = "./data"):
		assert which_split in set(["train", "test", "dev"])
		self.base_data_path = base_data_path
		self.filepath = os.path.join(self.base_data_path, "clean", "stsbenchmark", "sts-{}.csv".format(which_split))
		self._maybe_download_and_extract()

	def _maybe_download_and_extract(self):
		if not os.path.exists(self.filepath):
			raw_path = os.path.join(self.base_data_path, "raw")
			if not os.path.exists(raw_path):
				os.makedirs(raw_path)
			sts_path = download_sts(raw_path)
			sts_files = [f for f in os.listdir(sts_path) if f.endswith(".csv")]
			clean_sts(sts_files, sts_path, os.path.join(self.base_data_path, "clean", "stsbenchmark"))

	def as_dflow(self):
		return dp.auto_read_file(self.filepath).drop_columns('Column1')

	def as_dataframe(self):
		return dp.auto_read_file(self.filepath).drop_columns('Column1').to_pandas_dataframe()