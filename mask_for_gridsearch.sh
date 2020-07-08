#!/bin/bash
#SBATCH --cpus-per-task=1		# maximum cpu per task is 3.5 per gpus
#SBATCH --mem=3G	# 32G for larg job (40 holes and under)	# memory per node
#SBATCH --time=0-00:30			# time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Mask_for_gridsearch
#SBATCH --output=%x-%j.out
module load python/3.6
virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r cc_requirements.txt
pip install /home/aadam/scratch/ExoRIM/dist/ExoRim-0.1-py3-none-any.whl
python preprocessing/bispectra.py 
