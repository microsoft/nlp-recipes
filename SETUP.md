# Setup Guide

This document describes how to setup all the dependencies to run the notebooks in this repository.

The recommended environment to run these notebooks is the [Azure Data Science Virtual Machine (DSVM)](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/). Since a considerable number of the algorithms rely on deep learning, it is recommended to use a GPU DSVM.

For training at scale, operationalization or hyperparameter tuning, it is recommended to use [Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/service/).


## Table of Contents

* [Compute environments](#compute-environments)
* [Setup guide for Local or DSVM](#setup-guide-for-local-or-dsvm)
  * [Requirements](#requirements)
  * [Dependencies setup](#dependencies-setup)
  * [Register the conda environment in the DSVM JupyterHub](#register-the-conda-environment-in--the-dsvm-jupyterhub)


## Compute Environments

Depending on the type of NLP system and the notebook that needs to be run, there are different computational requirements. Currently, this repository supports **Python CPU** and **Python GPU**.


## Setup Guide for Local or DSVM

### Requirements

* A machine running Linux, MacOS or Windows.  
    > NOTE: Windows machines are not **FULLY SUPPORTED**. Please use at your own risk.  
* Miniconda or Anaconda with Python version >= 3.6. 
    * This is pre-installed on Azure DSVM such that one can run the following steps directly. To setup on your local machine, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a quick way to get started.
    * It is recommended to update conda to the latest version: `conda update -n base -c defaults conda`


### Dependencies Setup


We provide a script, [generate_conda_file.py](tools/generate_conda_file.py), to generate a conda-environment yaml file
which you can use to create the target environment using the Python version 3.6 with all the correct dependencies.

Assuming the repo is cloned as `nlp` in the system, to install **a default (Python CPU) environment**:

    cd nlp
    python tools/generate_conda_file.py
    conda env create -f nlp_cpu.yaml 

You can specify the environment name as well with the flag `-n`.

Click on the following menus to see how to install the Python GPU environment:

<details>
<summary><strong><em>Python GPU environment on Linux, MacOS</em></strong></summary>

Assuming that you have a GPU machine, to install the Python GPU environment, which by default installs the CPU environment:

    cd nlp
    python tools/generate_conda_file.py --gpu
    conda env create -n nlp_gpu -f nlp_gpu.yaml 

</details>

<details>
<summary><strong><em>Python GPU environment on Windows</em></strong></summary>

Assuming that you have an Azure GPU DSVM machine, here are the steps to setup the Python GPU environment:
1. Make sure you have CUDA Toolkit installed. You can run the command below in your terminal and you should see information printed out about your Cuda.

         nvcc --version
    
    If not, you can download the CUDA Toolkit and install from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

2. Create yaml file for enviroment setup. 

        cd nlp
        python tools/generate_conda_file.py --gpu
    
    Open the new file `nlp_gpu.yaml` created in your folder, check if the `cudatoolkit` package has consitant version as the CUDA Toolkit in your system. For example, if you download CUDA Toolkit 10.1, your `nlp_gpu.yaml` should have `cudatoolkit>=10`. Make necessary changes on the version and save the file.

3. Installs the GPU environment.

        conda env create -n nlp_gpu -f nlp_gpu.yaml 

</details>

### Register Conda Environment in DSVM JupyterHub

We can register our created conda environment to appear as a kernel in the Jupyter notebooks.

    conda activate my_env_name
    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"
    
If you are using the DSVM, you can [connect to JupyterHub](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro#jupyterhub-and-jupyterlab) by browsing to `https://your-vm-ip:8000`.  

## Install this repository via PIP
A [setup.py](setup.py) file is provied in order to simplify the installation of this utilities in this repo from the main directory. 

    pip install -e .

It is also possible to install directly from Github.

    pip install -e  git+git@github.com:microsoft/nlp.git@master#egg=utils_nlp

**NOTE** - The pip installation does not install any of the necessary package dependencies, it is expected that conda will be used as shown above to setup the environment for the utilities being used.