#!/bin/bash

### Install git-lfs ###
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install


### Install Python3.6 ###
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6 python3.6-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo python3.6 get-pip.py
sudo ln -s /usr/bin/python3.6 /usr/local/bin/python3
sudo ln -s /usr/local/bin/pip /usr/local/bin/pip3


### Create a clean Python3.6 environment ###
sudo pip3 install virtualenv
virtualenv -p /usr/bin/python3.6 .env
source ./.env/bin/activate


### Install dependencies ###
pip install -r requirements.txt


### Install submodules (SentEval) ###
git submodule init
git submodule update

sudo apt-get install unzip
cd SentEval/data/downstream/
./get_transfer_data.bash
