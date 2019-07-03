# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

from azureml.core import Workspace


def get_or_create_workspace(
    config_path=None,
    subscription_id=None,
    resource_group=None,
    workspace_name=None,
    workspace_region=None,
):
    """Get or create AzureML Workspace this will save the config to the path specified for later use

    Args:
        config_path (str): optional directory to look for / store config.json file (defaults to current directory)
        subscription_id (str): subscription id
        resource_group (str): resource group
        workspace_name (str): workspace name
        workspace_region (str): region

    Returns:
        Workspace
    """

    # use environment variables if needed
    if subscription_id is None:
        subscription_id = os.getenv("SUBSCRIPTION_ID")
    if resource_group is None:
        resource_group = os.getenv("RESOURCE_GROUP")
    if workspace_name is None:
        workspace_name = os.getenv("WORKSPACE_NAME")
    if workspace_region is None:
        workspace_region = os.getenv("WORKSPACE_REGION")

    # define fallback options in order to try
    options = [
        (
            Workspace,
            dict(
                subscription_id=subscription_id,
                resource_group=resource_group,
                workspace_name=workspace_name,
            ),
        ),
        (Workspace.from_config, dict(path=config_path)),
        (
            Workspace.create,
            dict(
                subscription_id=subscription_id,
                resource_group=resource_group,
                name=workspace_name,
                location=workspace_region,
                create_resource_group=True,
                exist_ok=True,
            ),
        ),
    ]

    for function, kwargs in options:
        try:
            ws = function(**kwargs)
            break
        except Exception:
            continue
    else:
        raise ValueError(
            "Failed to get or create AzureML Workspace with the configuration information provided"
        )

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
