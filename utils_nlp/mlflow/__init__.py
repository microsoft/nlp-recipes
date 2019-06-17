# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# TODO: Add tests

def set_mlflow_tracking_uri(tracking_uri=None, workspace=None, raise_on_error=False):
    """Set the mlflow tracking uri.
    Args:
        tracking_uri (str, optional): The desired tracking uri, calls
            mlflow.set_tracking_uri(tracking_uri). Defaults to the current working
            directory if other options no workspace is found.
        workspace (azureml.core.Workspace): An AzureML Workspace used to
            get the AzureML mlflow tracking uri. Defaults to the current
            workspace config if available.
        raise_on_error (bool, optional): Whether to raise if there are AzureML
            related failures. Defaults to False.
    """
    import mlflow
    if tracking_uri is not None and workspace is not None:
        raise ValueError("One of tracking_uri or workspace can be set. Not both.")
    elif tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        try:
            import azureml.mlflow
            workspace = workspace if workspace is not None else Workspace.from_config()
            mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
        except Exception:
            if raise_on_error:
                raise

            mlflow.set_tracking_uri(os.getcwd())
