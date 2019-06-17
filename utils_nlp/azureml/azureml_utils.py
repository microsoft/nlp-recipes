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
        azureml.core.Workspace
    """

    # use environment variables if needed
    if subscription_id is None:
        subscription_id = os.getenv("AZUREML_SUBSCRIPTION_ID")
    if resource_group is None:
        resource_group = os.getenv("AZUREML_RESOURCE_GROUP")
    if workspace_name is None:
        workspace_name = os.getenv("AZUREML_WORKSPACE_NAME")
    if workspace_region is None:
        workspace_region = os.getenv("AZUREML_WORKSPACE_REGION")

    # define fallback options in order to try
    try:
        ws = Workspace.from_config(path=config_path)
    except Exception:
        try:
            kwargs = dict(
                    subscription_id=subscription_id,
                    resource_group=resource_group,
                    name=workspace_name,
                    location=workspace_region,
                    create_resource_group=True,
                    exist_ok=True)
            ws = Workspace.create(**kwargs)
        except Exception:
            raise ValueError(
                "Failed to get or create AzureML Workspace with "
                "the configuration information provided: {}.".format(kwargs)
            )

    ws.write_config(path=config_path)
    return ws
