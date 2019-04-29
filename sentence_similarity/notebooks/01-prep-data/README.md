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
			<td rowspan=2>[SNLI](https://nlp.stanford.edu/projects/snli/)</td>
			<td>[snli_load.ipynb](snli_load.ipynb)</td>
			<td>Download and clean the SNLI dataset.</td>
		</tr>
		<tr>
			<td>[snli_preprocess.ipynb](snli_preprocess.ipynb)</td>
			<td>Lowercase, tokenize, and reshape the SNLI corpus for use in training the [Gensen](https://github.com/Maluuba/gensen) model. Demonstrates use of the [NLTK](https://www.nltk.org/) library for tokenization.</td>
		</tr>
		<tr>
			<td rowspan=2>[STS Benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)</td>
			<td>[sts_load.ipynb](sts_load.ipynb)</td>
			<td>Download and clean the STS Benchmark dataset.</td>
		</tr>
		<tr>
			<td>[sts_preprocess.ipynb](sts_load.ipynb)</td>
			<td>Lowercase and tokenize the STS Benchmark data using [spaCy](https://spacy.io/). Also shows how to remove stopwords from the tokens.</td>
		</tr>
		<tr>
			<td>[MSR Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398)</td>
			<td>[msrpc_load.ipynb](msrpc_load.ipynb)</td>
			<td>Download and clean the MSR Paraphrase corpus.</td>
		</tr>
	</tbody>
</table>