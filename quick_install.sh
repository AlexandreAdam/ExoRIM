#!/bin/sh

git clone https://github.com/AlexandreAdam/ExoRIM.git
cd ExoRIM
git checkout dev
conda create -n exorim_min python=3.6 -y
conda activate exorim_min
pip install -r minimal_requirements.txt
python setup.py develop
cd analysis
echo "Running martinache.py"
python martinache.py

