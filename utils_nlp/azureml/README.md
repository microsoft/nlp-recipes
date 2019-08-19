## [AzureML](.)

The AzureML submodule contains utilities to connect to a
[workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace),
train, tune and operationalize NLP systems at scale using AzureML.   
For example, the `DistributedCommunicator` class defined in
[azureml_bert_util.py](./azureml_bert_util.py) assists in making communication with multiple nodes
for distributed training possible. [azureml_utils.py](./azureml_utils.py) contains a few helper functions that make it easy to authenticate, create, or retrieve an AzureML resource.
