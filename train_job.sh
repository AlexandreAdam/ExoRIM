#!/bin/bash
#SBATCH --gres=gpu:2			# request GPU "generic resource"
#SBATCH --cpus-per-task=6		# maximum CPU on beluga=
#SBATCH --mem=32000M			# memory per node
#SBATCH --time=0-03:00			# time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=train_rim_with_bn_no_ph
#SBATCH --output=%x-%j.out
module load python/3.6
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r cc_requirements.txt
pip install /home/aadam/scratch/wheelhouse/celluloid-0.2.0-py3-none-any.whl
pip install /home/aadam/scratch/ExoRIM/dist/ExoRim-0.1-py3-none-any.whl
module load cuda cudnn 
python -W ignore::DeprecationWarning
python ExoRIM/trainv2.py 
