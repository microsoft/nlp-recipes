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
			<td rowspan=2>
				<a href="http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark">STS Benchmark</a>
			</td>
			<td>
				<a href="sts_load.ipynb">sts_load.ipynb</a>
			</td>
			<td>Download and clean the STS Benchmark dataset.</td>
		</tr>
		<tr>
			<td>
				<a href="sts_preprocess.ipynb">sts_load.ipynb</a>
			</td>
			<td>
				Lowercase and tokenize the STS Benchmark data using <a href="https://spacy.io/">spaCy</a>. Also shows how to remove stopwords from the tokens.
			</td>
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