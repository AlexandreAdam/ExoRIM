#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --account=def-lplevass
#SBATCH --time=1:0:0

source ~/environments/exorim3.8/bin/activate
python jwst_psf.py --filter F380M
