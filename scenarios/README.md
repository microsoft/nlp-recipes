# NLP Scenarios

This folder contains examples and best practices, written in Jupyter notebooks, for building Natural Language Processing systems for the following scenarios.


- [Text Classification](text_classification)
- [Named Entity Recognition](named_entity_recognition)
- [Entailment](entailment)
- [Question Answering](question_answering)
- [Sentence Similarity](sentence_similarity)
- [Embeddings](embeddings)
- [Annotation](annotation)

## Azure-enhanced notebooks

Azure products and services are used in certain notebooks to enhance the efficiency of developing Natural Language systems at scale.

To successfully run these notebooks, the users **need an Azure subscription** or can [use Azure for free](https://azure.microsoft.com/en-us/free/).

The Azure products featured in the notebooks include:

* [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) - Azure Machine Learning service is a cloud service used to train, deploy, automate, and manage machine learning models, all at the broad scale that the cloud provides. It is used across various notebooks for the AI model development related tasks like:
  * Using Datastores
  * Tracking and monitoring metrics to enhance the model creation process
  * Distributed Training
  * Hyperparameter tuning
  * Scaling up and out on Azure Machine Learning Compute
  * Deploying a web service to both Azure Container Instance and Azure Kubernetes Service

* [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where#aks) - You can use Azure Machine Learning service to host your model in a web service deployment on Azure Kubernetes Service (AKS). AKS is good for high-scale production deployments and provides autoscaling, and fast response times.

* [Azure Container Instance](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where#aci)- You can use Azure Machine Learning service to host your model in a web service deployment on Azure Container Instance (ACI). ACI is good for low scale, CPU-based workloads.

There may be other Azure service or products used in the notebooks. Introduction and/or reference of those will be provided in the notebooks.
