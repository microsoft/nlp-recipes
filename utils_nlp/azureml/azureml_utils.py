# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.authentication import AuthenticationException
from azureml.core import Workspace

def get_auth():
    """ 
    Method to get the correct Azure ML Authentication type
    
    Always start with CLI Authentication and if it fails, fall back 
    to interactive login
    """
    try:
        auth_type = AzureCliAuthentication()
        auth_type.get_authentication_header()
    except AuthenticationException:
        auth_type = InteractiveLoginAuthentication()
    return auth_type


def get_or_create_workspace(
    config_path = None,
    subscription_id = None,
    resource_group = None,
    workspace_name = None,
    workspace_region = None
    ) -> Workspace:
    """
    Returns workspace if one exists already with the name
    otherwise creates a new one.

    Args
    config_path: optional directory to look for / store config.json file (defaults to current directory)
    subscription_id: Azure subscription id
    resource_group: Azure resource group to create workspace and related resources  
    workspace_name: name of azure ml workspace  
    workspace_region: region for workspace 
    """
    try:
        # get existing azure ml workspace
        ws = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            auth=get_auth(),
        )

    except:
        # this call might take a minute or two.
        print("Creating new workspace")
        ws = Workspace.create(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            create_resource_group=True,
            location=workspace_region,
            auth=get_auth()
            )

    print(config_path)
    ws.write_config(path=config_path)
    return ws


def log_metrics_scalar(value, run, name="", description=None):
    """Log scalar metric to the AzureML run
    
    Args:
        value : numerical or string value to log
        run : AzureML Run object
        name : name of metric
        description : description of metric
    """
    run.log(name, value, description)

def log_metrics_table(df, run, name="", description=None, as_scalar=False):
    """Log data from pd.DataFrame to the AzureML run
    
    Args:
        df : pd.DataFrame containing metrics to log
        run : AzureML Run object
        name : name of metric
        description : description of metric
        as_scalar : when True, logs each cell of the table as a scalar metric; defaults to False
    """
    if as_scalar:
        for rn in df.index:
            for cn in df.columns:
                log_metrics_scalar(df.loc[rn, cn], run, name="{0}::{1}".format(rn, cn), description=description)

    else:
        run.log_table(name, df.to_dict(), description)
