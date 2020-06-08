#!/bin/bash
#SBATCH --gres=gpu:1			# request GPU "generic resource"
#SBATCH --cpus-per-task=3		# maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32000M			# memory per node
#SBATCH --time=0-05:00			# time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Train_ExoRIM
#SBATCH --output=%x-%j.out
module load python/3.6
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r cc_requirements.txt
pip install /home/aadam/scratch/ExoRIM/dist/ExoRim-0.1-py3-none-any.whl
module load cuda cudnn 
python -W ignore::DeprecationWarning
python main.py -n 1000 -s 0.8 -t 4 --holes 20 -p 20 -e 1000 -b 100 --out_save_mod 100
