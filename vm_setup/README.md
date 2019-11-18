# Setup Multiple Virtual Machines with Multiple JupyterHub User Accounts

This shows an example of deploying multiple Azure VM (Virtual Machine) with multiple JupyterHub user accounts.

To make the deployment and management easy, we utilize Azure VMSS (Virtual Machine Scale Set) with DSVM (Data Science Virtual Machine) image. After the VMSS deployment, we invoke a shell script on each VM instance to create multiple JupyterHub user accounts with a desired conda environment.