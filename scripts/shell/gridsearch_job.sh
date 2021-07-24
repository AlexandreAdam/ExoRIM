#!/bin/bash
#SBATCH --array=1-10%1                  # make an array of 10 job like this one to be executed in parallel
#SBATCH --gres=gpu:1			# request GPU "generic resource"
#SBATCH --cpus-per-task=3		# maximum cpu per task is 3.5 per gpus
#SBATCH --mem=16G			# memory per node
#SBATCH --time=0-10:00			# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Gridsearch_ExoRIM
#SBATCH --output=%x-%j.out
source $HOME/environments/exorim3.8/bin/activate
module load cuda cudnn 
python -W ignore::DeprecationWarning
python gridsearch.py -n 1000 -t 1 -e 100 -b 25 --model_trained 100
