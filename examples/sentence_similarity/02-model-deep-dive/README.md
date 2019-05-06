# Model Select and Optimize

In this directory, notebooks are provided to demonstrate how to train, tune and 
optimize hyperparameters of sentence 
similarity models with the 
utility functions([nlp_utils](../../../nlp_utils))  and Azure Machine Learning 
service. 

| Notebook | Description | 
| --- | --- | 
| [tuning_spark_als](train-gensen-with-distributed pytorch-on-AML.ipynb) | Step by step tutorials on how to train Gensen using distributed pytorch on AzureML Compute.

### Prerequisites
To run the examples running on the Azure Machine Learning service, the 
[`azureml-sdk`](https://pypi.org/project/azureml-sdk/) is required. The AzureML 
Python SDK is already installed after setting up the conda environments from this 
repository (see [environment.yml](../../../environment.yml)). 

More info about setting up an AzureML environment can be found at [this link](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment).

### AzureML Workspace Configuration
AzureML workspace is the foundational block in the cloud that you use to experiment, train, and deploy machine learning models. We 
1. set up a workspace from Azure portal and 
2. create a config file manually. 

The instructions here are based on AzureML documents about [Quickstart with Azure portal](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-get-started) and [Quickstart with Python SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python) where you can find more details with screenshots about the setup process.
  
#### Create a workspace
1. Sign in to the [Azure portal](https://portal.azure.com) by using the credentials for the Azure subscription you use.
2. Select **Create a resource** menu, search for **Machine Learning service workspace** select **Create** button.
3. In the **ML service workspace** pane, configure your workspace by entering the *workspace name* and *resource group* (or **create new** resource group if you don't have one already), and select **Create**. It can take a few moments to create the workspace.
  
#### Make a configuration file
Create a *./aml_config/config.json* file with the following contents:
```
{
    "subscription_id": "<subscription-id>",
    "resource_group": "<resource-group>",
    "workspace_name": "<workspace-name>"
}
```
