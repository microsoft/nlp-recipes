# Examples

This folder contains examples and best practices, written in Jupyter notebooks, for building Natural Language Processing systems for the following scenarios.

|Category|Applications|Methods|Languages|
|---| ------------------------ | ------------------- |---|
|[Text Classification](text_classification)|Topic Classification|BERT, XLNet, RoBERTa, DistilBERT|en, hi, ar|
|[Named Entity Recognition](named_entity_recognition) |Wikipedia NER|BERT|en|
|[Text Summarization](text_summarization)|News Summarization, Headline Generation|Extractive: BERTSumExt <br> Abstractive: UniLM (s2s-ft)|en
|[Entailment](entailment)|MultiNLI Natural Language Inference|BERT|en|
|[Question Answering](question_answering) |SQuAD|BiDAF, BERT, XLNet, DistilBERT|en|
|[Sentence Similarity](sentence_similarity)|STS Benchmark|BERT, GenSen|en|
|[Embeddings](embeddings)|Custom Embeddings Training|Word2Vec, fastText, GloVe|en|
|[Annotation](annotation)|Text Annotation|Doccano|en|
|[Model Explainability](model_explainability)|DNN Layer Explanation|DUUDNM (Guan et al.)|en|

## Data/Telemetry
The Azure Machine Learning notebooks collect browser usage data and send it to Microsoft to help improve our products and services. Read Microsoft's [privacy statement to learn more](https://privacy.microsoft.com/en-US/privacystatement).

To opt out of tracking, a Python [script](../tools/remove_pixelserver.py) under the `tools` folder is also provided. Executing the script will check all notebooks under the `examples` folder, and automatically remove the telemetry cell:

```sh
python ../tools/remove_pixelserver.py
```
