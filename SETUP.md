# Setup guide

This document describes how to setup all the dependencies to run the notebooks in this repository.

The recommended environment to run these notebooks is the [Azure Data Science Virtual Machine (DSVM)](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/). Since a considerable number of the algorithms rely on deep learning, it is recommended to use a GPU DSVM.

For training at scale, operationalization or hyperparameter tuning, it is recommended to use [Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/service/).


## Table of Contents

* [Compute environments](#compute-environments)
* [Setup guide for Local or DSVM](#setup-guide-for-local-or-dsvm)
  * [Setup Requirements](#setup-requirements)
  * [Dependencies setup](#dependencies-setup)
  * [Register the conda environment in the DSVM JupyterHub](#register-the-conda-environment-in--the-dsvm-jupyterhub)


## Compute environments

Depending on the type of NLP system and the notebook that needs to be run, there are different computational requirements.

Currently, this repository supports the following environments:

* Python CPU
* Python GPU


## Setup guide for Local or DSVM

### Setup Requirements

* Anaconda with Python version >= 3.6. [Miniconda](https://conda.io/miniconda.html) is the fastest way to get started.
* The Python library dependencies can be found in this [script](tools/generate_conda_file.sh).

### Dependencies setup


We provide a script to [generate a conda file](tools/generate_conda_file.sh), depending of the environment we want to use. This will create the environment using the Python version 3.6 with all the correct dependencies.

To install each environment, first we need to generate a conda yaml file and then install the environment. We can specify the environment name with the input `-n`.

Click on the following menus to see more details:

<details>
<summary><strong><em>Python CPU environment</em></strong></summary>

Assuming the repo is cloned as `NLP` in the system, to install the Python CPU environment:

    cd NLP
    ./tools/generate_conda_file.sh
    conda env create -n nlp_cpu -f nlp_cpu.yaml 

</details>

<details>
<summary><strong><em>Python GPU environment</em></strong></summary>

Assuming that you have a GPU machine, to install the Python GPU environment, which by default installs the CPU environment:

    cd NLP
    ./tools/generate_conda_file.sh --gpu
    conda env create -n nlp_gpu -f nlp_gpu.yaml 

</details>



### Register the conda environment in the DSVM JupyterHub

DSVM comes with a preinstalled JupyterHub, which is accessible through port 8000. To access it just type in your browser `https://your-vm-ip:8000`. See more details [in this tutorial](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro#jupyterhub-and-jupyterlab).

When using the DSVM, we can register our created conda environment to appear as a kernel in JupyterHub. 

    conda activate my_env_name
    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"

