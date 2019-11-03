# Aspect Based Sentiment Analysis

This folder contains examples and best practices, written in Jupyter notebooks, for training [Aspect Based Sentiment Analysis Models using Intel's NLP Architect](http://nlp_architect.nervanasys.com/absa.html)
 models with the azure machine learning service.

# What is Aspect Based Sentiment Analysis?

Aspect based sentiment analysis (ABSA) is an advanced sentiment analysis technique that identifies and provides coresponding sentiment scores to the aspects of a given text. ABSA a powerful tool for getting actionable insight from your text data.

For example consider the sentence following resturant review 

```
The ambiance is charming. Uncharacteristically, the service was DREADFUL.When we wanted to pay our bill at the end of the evening, our waitress was nowhere to be found...
```

While traditional sentiment analysis models such as [Azure Text Analytics](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/?WT.mc_id=absa-notebook-abornst) will correctly classify the sentiment of this model as negative. An aspect based model will provide more granular insight by highlighting the fact that the while the **service** and **waitress** provided a negative expirence the resturants **ambiance** was indeed positive.

## Summary

|Notebook|Environment|Description|Dataset|
|---|---|---|---|
|[Aspect based sentiment analysis](absa.ipynb)|Local| A notebook for training and deploying [Aspect Based Sentiment Analysis Models using Intel's NLP Architect](http://nlp_architect.nervanasys.com/absa.html) |
