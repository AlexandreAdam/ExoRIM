#!/bin/bash
#SBATCH --gres=gpu:1			# request GPU "generic resource"
#SBATCH --cpus-per-task=3		# maximum cpu per task is 3.5 per gpus
#SBATCH --mem=4G			# memory per node
#SBATCH --time=0-5:00			# time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Train_ExoRIM
#SBATCH --output=%x-%j.out
module load python/3.6
virtualenv $SLURM_TMPDIR/envc
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r compute_canada_requirements.txt
pip install /home/aadam/scratch/ExoRIM/dist/ExoRim-0.1-py3-none-any.whl
module load cuda cudnn 
python -W ignore::DeprecationWarning
python ../main.py -n 1000 -s 0.8 -t 4 -p 10 -e 50 -b 25 --index_save_mod 250 --SNR 100
