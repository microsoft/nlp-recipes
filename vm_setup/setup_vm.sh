cd /home/ 

# clone repo and install the conda env 
git clone https://github.com/microsoft/nlp-recipes.git

# change permission as we copy this into each user's folder
chmod -R ugo+rwx /home/nlp-recipes

source /data/anaconda/etc/profile.d/conda.sh

cd /home/nlp-recipes
python tools/generate_conda_file.py
conda env create -n nlp_cpu -f nlp_cpu.yaml

conda activate nlp_cpu
python -m ipykernel install --name nlp_cpu --display-name "Python (nlp_cpu)"

# Pip install the utils_nlp package in the conda environment
pip install -e . 


# add users to jupyterhub
echo 'c.Authenticator.whitelist = {"mlads1", "mlads2", "mlads3", "mlads4", "mlads5", "mlads6", "mlads7", "mlads8", "mlads9", "mlads10", "mlads11", "mlads12", "mlads13", "mlads14", "mlads15"}' | sudo tee -a /etc/jupyterhub/jupyterhub_config.py

# create the users on the vm 
for i in {1..15}
do
    USERNAME=mlads$i
    PASSWORD=nlpmlads$i
    sudo adduser --quiet --disabled-password --gecos "" $USERNAME
    echo "$USERNAME:$PASSWORD" | sudo chpasswd
    rm -rf /data/home/$USERNAME/notebooks/*
    
    # copy repo
    cp -ar /home/nlp-recipes /data/home/$USERNAME/notebooks
done

# restart jupyterhub 
sudo systemctl stop jupyterhub 
sudo systemctl start jupyterhub 

exit
