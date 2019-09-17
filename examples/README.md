# Examples

This folder contains examples and best practices, written in Jupyter notebooks, for building Natural Language Processing systems for the following scenarios.

|Category|Applications|Methods|Languages|
|---| ------------------------ | ------------------- |---|
|[Text Classification](text_classification)|Topic Classification|BERT, XLNet|en, hi, ar|
|[Named Entity Recognition](named_entity_recognition) |Wikipedia NER|BERT|en|
|[Entailment](entailment)|MultiNLI Natural Language Inference|BERT|en|
|[Question Answering](question_answering) |SQuAD|BiDAF, BERT|en|
|[Sentence Similarity](sentence_similarity)|STS Benchmark|Representation: TF-IDF, Word Embeddings, Doc Embeddings<br>Metrics: Cosine Similarity, Word Mover's Distance<br> Models: BERT, GenSen||
|[Embeddings](embeddings)|Custom Embeddings Training|Word2Vec, fastText, GloVe||
|[Annotation](annotation)|Text Annotation|Doccano||
|[Model Explainability](model_explainability)|DNN Layer Explanation|DUUDNM (Guan et al.)|

## Data/Telemetry
The Azure Machine Learning notebooks collect browser usage data and send it to Microsoft to help improve our products and services. Read Microsoft's [privacy statement to learn more](https://privacy.microsoft.com/en-US/privacystatement).

To opt out of tracking, please go to the raw `.ipynb` files and remove the following line of code (the URL will be slightly different depending on the file):

```sh
    "![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/nlp/examples/text_classification/tc_bert_azureml.png)"
```