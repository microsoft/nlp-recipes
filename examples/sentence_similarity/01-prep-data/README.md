# Data Preparation

<table>
	<thead>
		<tr>
			<th>Dataset</th>
			<th>Notebook</th>
			<th>Description</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td rowspan=2>
				<a href="https://nlp.stanford.edu/projects/snli/">SNLI</a>
			</td>
			<td>
				<a href="snli_load.ipynb">snli_load.ipynb</a>
			</td>
			<td>
				Download and clean the SNLI dataset.
			</td>
		</tr>
		<tr>
			<td>
				<a href="snli_preprocess.ipynb">snli_preprocess.ipynb</a>
			</td>
			<td>
				Lowercase, tokenize, and reshape the SNLI corpus for use in training the <a href="https://github.com/Maluuba/gensen">Gensen</a> model. Demonstrates use of the <a href="https://www.nltk.org/">NLTK</a> library for tokenization.
			</td>
		</tr>
		<tr>
			<td>
				<a href="http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark">STS Benchmark</a>
			</td>
			<td>
				<a href="stsbenchmark.ipynb">sts_load.ipynb</a>
			</td>
			<td>Downloads and cleans the STS Benchmark dataset. Shows an example of tokenizing and removing stopwords using the popular <a href="https://spacy.io/">spaCy</a> library.</td>
		</tr>
		<tr>
			<td>
				<a href="https://www.microsoft.com/en-us/download/details.aspx?id=52398">MSR Paraphrase Corpus</a>
			</td>
			<td>
				<a href="msrpc_load.ipynb">msrpc_load.ipynb</a>
			</td>
			<td>Download and clean the MSR Paraphrase corpus.</td>
		</tr>
	</tbody>
</table>