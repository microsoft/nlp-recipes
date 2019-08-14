### [AzureML](azureml)

The AzureML submodule contains utilities to connect to a [workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace), train, tune and operationalize NLP systems at scale using AzureML. 

It contains the following scripts:  

1. [azureml_bert_util.py](./azureml_bert_util.py) - This script defines a class `DistributedCommunicator` which utilizes `torch`. It defines functionalities that assist in making communication with multiple nodes for distributed training possible. It assists in registering hooks for your model, create reduction tensors and synchronize processes for an AzureML cluster.   

1. [azureml_utils.py](./azureml_utils.py) - This script defines a functionalities that makes it easy to authenticate, create an Azure ML workspace or retrieve an existing one, and create the compute target or retrieves one if exists.   

See example usage below for how to connect to a workspace. 


```python
from utils_nlp.azureml.azureml_utils import get_or_create_workspace

###Note: you do not need to fill in these values if you have a config.json in the same folder as this notebook
ws = get_or_create_workspace(
    config_path=config_path,
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name,
    workspace_region=workspace_region,
)
```  
 