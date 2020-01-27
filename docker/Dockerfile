FROM nvidia/cuda

# Install Anaconda
# Non interactive installation instructions can be found 
# https://hub.docker.com/r/continuumio/anaconda/dockerfile
# https://hub.docker.com/r/continuumio/miniconda/dockerfile
ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Get the latest version repository
WORKDIR /root
RUN apt-get install -y zip && \
    wget --quiet https://github.com/microsoft/nlp-recipes/archive/staging.zip -O staging.zip && \
    unzip staging.zip  && rm staging.zip 
    
# Install the packages
WORKDIR /root/nlp-recipes-staging
RUN python /root/nlp-recipes-staging/tools/generate_conda_file.py --gpu && \
    conda env create -n nlp_gpu -f nlp_gpu.yaml 
RUN source activate nlp_gpu && \
    pip install -e . && \
    python -m ipykernel install --user --name nlp_gpu --display-name "Python (nlp_gpu)"

# Run notebook
EXPOSE 8888/tcp
WORKDIR /root/nlp-recipes-staging
CMD source activate nlp_gpu && \
    jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser --notebook-dir .
