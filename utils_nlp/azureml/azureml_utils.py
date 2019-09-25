# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Helper functions for interacting with AzureML Resources."""

import os
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.authentication import AuthenticationException
from azureml.core import Workspace
from azureml.exceptions import ProjectSystemException
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


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
    config_path="./.azureml",
    subscription_id=None,
    resource_group=None,
    workspace_name=None,
    workspace_region=None,
):
    """
    Method to get or create workspace.

    Args:
        config_path: optional directory to look for / store config.json file (defaults to current
            directory)
        subscription_id: Azure subscription id
        resource_group: Azure resource group to create workspace and related resources
        workspace_name: name of azure ml workspace
        workspace_region: region for workspace

    Returns:
        obj: AzureML workspace if one exists already with the name otherwise creates a new one.
    """
    config_file_path = "."

    if config_path is not None:
        config_dir, config_file_name = os.path.split(config_path)
        if config_file_name != "config.json":
            config_file_path = os.path.join(config_path, "config.json")

    try:
        # get existing azure ml workspace
        if os.path.isfile(config_file_path):
            ws = Workspace.from_config(config_file_path, auth=get_auth())
        else:
            ws = Workspace.get(
                name=workspace_name,
                subscription_id=subscription_id,
                resource_group=resource_group,
                auth=get_auth(),
            )

    except ProjectSystemException:
        # this call might take a minute or two.
        print("Creating new workspace")
        ws = Workspace.create(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            create_resource_group=True,
            location=workspace_region,
            auth=get_auth(),
        )

        ws.write_config(path=config_path)
    return ws


def get_or_create_amlcompute(
    workspace,
    compute_name,
    vm_size="",
    min_nodes=0,
    max_nodes=None,
    idle_seconds_before_scaledown=None,
    verbose=False,
):
    """
        Get or create AmlCompute as the compute target. If a cluster of the same name is found,
        attach it and rescale accordingly. Otherwise, create a new cluster.

    Args:
        workspace (Workspace): workspace
        compute_name (str): name
        vm_size (str, optional): vm size
        min_nodes (int, optional): minimum number of nodes in cluster
        max_nodes (None, optional): maximum number of nodes in cluster
        idle_seconds_before_scaledown (None, optional): how long to wait before the cluster
            autoscales down
        verbose (bool, optional): if true, print logs
    Returns:
        Compute target
    """
    try:
        if verbose:
            print("Found compute target: {}".format(compute_name))

        compute_target = ComputeTarget(workspace=workspace, name=compute_name)
        if len(compute_target.list_nodes()) < max_nodes:
            if verbose:
                print("Rescaling to {} nodes".format(max_nodes))
            compute_target.update(max_nodes=max_nodes)
            compute_target.wait_for_completion(show_output=verbose)

    except ComputeTargetException:
        if verbose:
            print("Creating new compute target: {}".format(compute_name))

        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            idle_seconds_before_scaledown=idle_seconds_before_scaledown,
        )
        compute_target = ComputeTarget.create(workspace, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=verbose)

    return compute_target


def get_output_files(run, output_path, file_names=None):
    """
    Method to get the output files from an AzureML output directory.

    Args:
        file_names(list): Names of the files to download.
        run(azureml.core.run.Run): Run object of the run.
        output_path(str): Path to download the output files.

    Returns: None

    """
    os.makedirs(output_path, exist_ok=True)

    if file_names is None:
        file_names = run.get_file_names()

    for f in file_names:
        dest = os.path.join(output_path, f.split("/")[-1])
        print("Downloading file {} to {}...".format(f, dest))
        run.download_file(f, dest)
