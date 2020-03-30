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
* [Setup guide for docker](#Set-up-guide-for-nvidia-docker)

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
<summary><strong><em>Python GPU environment</em></strong></summary>

Assuming that you have a GPU machine, to install the Python GPU environment, 
1. Check the CUDA **driver** version on your machine by running

        nvidia-smi
    The top of the output shows the CUDA **driver** version, which is 10.0 in the example below.   
    +-----------------------------------------------------------------------------+  
    | NVIDIA-SMI 410.79 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Driver Version: 410. &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;CUDA Version: 10.0     |  
    |-------------------------------+----------------------+----------------------+
2. Decide which cuda **runtime** version you should install.   
The cuda **runtime** version is the version of the cudatoolkit that will be installed in the conda environment in the next step, which should be <= the CUDA **driver** version found in step 1.  
Currently, this repo uses PyTorch 1.4.0 which is compatible with cuda 9.2 and cuda 10.1. The conda environment file generated in step 3 installs cudatoolkit 10.1 by default. If your CUDA **driver** version is < 10.1, you should add additional argument "--cuda_version 9.2" when calling generate_conda_files.py.   

3. Install the GPU environment:  
If CUDA **driver** version >= 10.1

        cd nlp-recipes
        python tools/generate_conda_file.py --gpu
        conda env create -n nlp_gpu -f nlp_gpu.yaml

    If CUDA **driver** version < 10.1

        cd nlp-recipes
        python tools/generate_conda_file.py --gpu --cuda_version 9.2
        conda env create -n nlp_gpu -f nlp_gpu.yaml

4. Enable mixed precision training (optional)  
Mixed precision training is particularly useful if your model takes a long time to train. It usually reduces the training time by 50% and produces the same model quality. To enable mixed precision training, run the following command 

        conda activate nlp_gpu
        git clone https://github.com/NVIDIA/apex.git
        cd apex
        pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

    **Troubleshooting**:  
    If you run into an error message "RuntimeError: Cuda extensions are being compiled with a version of Cuda that does not match the version used to compile Pytorch binaries.", you need to make sure your NVIDIA Cuda compiler driver (nvcc) version and your cuda **runtime** version are exactly the same. To check the nvcc version, run   

        nvcc -V

    If the nvcc version is 10.0, it's recommended to upgrade to 10.1 and re-create your conda environment with cudatoolkit=10.1.
    
    **Steps to upgrade CUDA **driver** version and nvcc version**  
    We have tested the following steps. Alternatively, you can follow the official instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)  
    a. Update apt-get and reboot your machine

        sudo apt-get update
        sudo apt-get upgrade --fix-missing
        sudo reboot
    b. Download the CUDA toolkit .run file from https://developer.nvidia.com/cuda-10.1-download-archive-base based on your target platform. For example, on a Linux machine with Ubuntu 16.04, run   

        wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run  

    c. Upgrade CUDA driver by running  

        sudo sh cuda_10.1.105_418.39_linux.run
    First, accept the user agreement.  
    ![](https://nlpbp.blob.core.windows.net/images/upgrade_cuda_driver/1agree_to_user_agreement.PNG)  
    Next, choose the components to install.  
    It's possible that you already have NVIDIA driver 418.39 and CUDA 10.1, but nvcc 10.0. In this case, you can uncheck the "DRIVER" box and upgrade nvcc by re-installing CUDA toolkit only.   
    ![](https://nlpbp.blob.core.windows.net/images/upgrade_cuda_driver/2install_cuda_only.PNG)  

    If you choose to install all components, follow the instructions on the screen to uninstall existing NVIDIA driver and CUDA toolkit first.  
    ![](https://nlpbp.blob.core.windows.net/images/upgrade_cuda_driver/3install_all.PNG)   
    Then re-run   

        sudo sh cuda_10.1.105_418.39_linux.run
    Select "Yes" to update the cuda symlink.   
    ![](https://nlpbp.blob.core.windows.net/images/upgrade_cuda_driver/4Upgrade_symlink.PNG)  

    d. Run the following commands again to make sure you have NVIDIA driver 418.39, CUDA driver 10.1 and nvcc 10.1

        nvidia-smi
        nvcc -V

    e. Repeat steps 3 & 4 to recreate your conda environment with cudatoolkit **runtime** 10.1 and apex installed for mixed precision training. 


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

# Set up guide for (nvidia) docker

## Pre-requisites
In order to use the notebooks within a docker enviornment, you will need to have [nvidia docker drivers](https://github.com/NVIDIA/nvidia-docker) and [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) installed on your computer.

## Building docker image
A docker file is provided within the [docker](docker/) folder. You can create the image using 
```
  cd docker
  docker build -f . -t nlp-recipes
```
This will create a docker image containing all the dependencies and will name it as nlp-recipies:latest

## Running the container
You can run the notebook within the container environment using
```
  docker run --gpus all -p 8888:8888 nlp-recipes
```
This will map port 8888 of the local machine 

## Trouble shooting
* If you have permission issues with `docker build` or `docker run`, you might need to run docker with sudo permissions. 
* If you are getting 'port already in use' errors, consider mapping a different port on the local machine to port 8888 on the container e.g.
```
docker run --gpus all -p 9000:8888 nlp-recipes
```
