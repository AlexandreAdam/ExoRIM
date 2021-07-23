#!/bin/bash
#SBATCH --gres=gpu:1			# request GPU "generic resource"
#SBATCH --cpus-per-task=3		# maximum cpu per task is 3.5 per gpus
#SBATCH --mem=16G			# memory per node
#SBATCH --time=0-10:00			# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_ExoRIM
#SBATCH --output=%x-%j.out
#module load python/3.6
#virtualenv $SLURM_TMPDIR/envc
#source $SLURM_TMPDIR/env/bin/activate
#pip install --no-index --upgrade pip
#pip install --no-index -r compute_canada_requirements.txt
#pip install /home/aadam/scratch/ExoRIM/dist/ExoRim-0.1-py3-none-any.whl
source $HOME/environments/exorim3.8/bin/activate
module load cuda cudnn 
python -W ignore::DeprecationWarning
python main.py -n 1000 -t 10 -p 50 -e 500 -b 25 --SNR=100 --infinite_dataset --checkpoint=100 --pixels=64 --learning_rate=1e-4 --hparams_json="best_hparams.json"
