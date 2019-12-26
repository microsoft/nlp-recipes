# Setup Guide

This document describes how to setup all the dependencies to run the notebooks in this repository.

The recommended environment to run these notebooks is the [Azure Data Science Virtual Machine (DSVM)](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/). Since a considerable number of the algorithms rely on deep learning, it is recommended to use a GPU DSVM.

For training at scale, operationalization or hyperparameter tuning, it is recommended to use [Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/service/).


## Table of Contents

* [Compute environments](#compute-environments)
* [Create a cloud-based workstation (Optional)](#Create-a-cloud-based-workstation-optional)
* [Setup guide for Local or Virtual Machines](#setup-guide-for-local-or-virtual-machines)
  * [Requirements](#requirements)
  * [Dependencies setup](#dependencies-setup)
  * [Register the conda environment in the DSVM JupyterHub](#register-conda-environment-in-dsvm-jupyterhub)
  * [Installing the Repo's Utils via PIP](#installing-the-repos-utils-via-pip)


## Compute Environments

Depending on the type of NLP system and the notebook that needs to be run, there are different computational requirements. Currently, this repository supports **Python CPU** and **Python GPU**. A conda environment YAML file can be generated for either CPU or GPU environments as shown below in the *Dependencies Setup* section.

## Create a cloud-based workstation (Optional)

[Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/)’s Notebook Virtual Machine (VM), is a cloud-based workstation created specifically for data scientists. Notebook VM based authoring is directly integrated into Azure Machine Learning service, providing a code-first experience for Python developers to conveniently build and deploy models in the workspace. Developers and data scientists can perform every operation supported by the Azure Machine Learning Python SDK using a familiar Jupyter notebook in a secure, enterprise-ready environment. Notebook VM is secure and easy-to-use, preconfigured for machine learning, and fully customizable. 

You can learn how to create a Notebook VM [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-1st-experiment-sdk-setup#azure) and then follow the same setup as in the [Setup guide for Local or DSVM](#setup-guide-for-local-or-dsvm-machines) directly using the terminal in the Notebook VM.

## Setup Guide for Local or Virtual Machines

### Requirements

* A machine running Linux, MacOS or Windows.
* On Windows, Microsoft Visual C++ 14.0 is required for building certain packages. Download Microsoft Visual C++ Build Tools [here](https://visualstudio.microsoft.com/downloads/).

* Miniconda or Anaconda with Python version >= 3.6.
    * This is pre-installed on Azure DSVM such that one can run the following steps directly. To setup on your local machine, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a quick way to get started.
    * It is recommended to update conda to the latest version: `conda update -n base -c defaults conda`

> NOTE: Windows machines are not **FULLY SUPPORTED**. Please use at your own risk.

### Dependencies Setup


We provide a script, [generate_conda_file.py](tools/generate_conda_file.py), to generate a conda-environment yaml file
which you can use to create the target environment using the Python version 3.6 with all the correct dependencies.

Assuming the repo is cloned as `nlp-recipes` in the system, to install **a default (Python CPU) environment**:

    cd nlp-recipes
    python tools/generate_conda_file.py
    conda env create -f nlp_cpu.yaml

You can specify the environment name as well with the flag `-n`.

Click on the following menus to see how to install the Python GPU environment:

<details>
<summary><strong><em>Python GPU environment on Linux, MacOS</em></strong></summary>

Assuming that you have a GPU machine, to install the Python GPU environment, which by default installs the CPU environment:

    cd nlp-recipes
    python tools/generate_conda_file.py --gpu
    conda env create -n nlp_gpu -f nlp_gpu.yaml

</details>

<details>
<summary><strong><em>Python GPU environment on Windows</em></strong></summary>

Assuming that you have an Azure GPU DSVM machine, here are the steps to setup the Python GPU environment:
1. Make sure you have CUDA Toolkit version 9.0 above installed on your Windows machine. You can run the command below in your terminal to check.

         nvcc --version
    If you don't have CUDA Toolkit or don't have the right version, please download it from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

2. Install the GPU environment.

        cd nlp-recipes
        python tools/generate_conda_file.py --gpu
        conda env create -n nlp_gpu -f nlp_gpu.yaml

</details>

### Register Conda Environment in DSVM JupyterHub

We can register our created conda environment to appear as a kernel in the Jupyter notebooks.

    conda activate my_env_name
    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"

If you are using the DSVM, you can [connect to JupyterHub](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro#jupyterhub-and-jupyterlab) by browsing to `https://your-vm-ip:8000`.  If you are prompted to enter user name and password, enter the user name and password that you use to log in to your virtual machine. 

### Installing the Repo's Utils via PIP

<details>
    <summary>The utils_nlp module of this repository needs to be installed as a python package in order to be used by the examples. <strong><em>Click to expand and see the details</em></strong> 
    </summary> 
    <p>  
A setup.py file is provided in order to simplify the installation of this utilities in this repo from the main directory.  
    
To install the package, please run the command below (from directory root)

    pip install -e . 

Running the command tells pip to install the `utils_nlp` package from source in [development mode](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode). This just means that any updates to `utils_nlp` source directory will immediately be reflected in the installed package without needing to reinstall; a very useful practice for a package with constant updates.   

> It is also possible to install directly from Github, which is the best way to utilize the `utils_nlp` package in external projects (while still reflecting updates to the source as it's installed as an editable `'-e'` package). 

>   `pip install -e  git+git@github.com:microsoft/nlp-recipes.git@master#egg=utils_nlp`  

Either command, from above, makes `utils_nlp` available in your conda virtual environment. You can verify it was properly installed by running:  

    pip list  
    

**NOTE** - The pip installation does not install any of the necessary package dependencies, it is expected that conda will be used as shown above to setup the environment for the utilities being used.
    </p>
</details>

The details of the versioning info can be found at [VERSIONING.md](VERSIONING.md).

