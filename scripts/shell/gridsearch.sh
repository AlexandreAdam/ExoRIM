#!/bin/bash
#SBATCH --array=1-32
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-10:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Gridsearch_exorim
#SBATCH --output=%x-%j.out
source $HOME/environments/exorim3.8/bin/activate
python $EXORIM_PATH/scripts/gridsearch.py\
  --n_models=32\
  --max_time=9.5\
  --strategy=uniform\
  --total_items 1000\
  --width=2\
  --batch_size 1 10\
  --steps 6 8 10\
  --log_floor 1e-4\
  --filters 64 128 256\
  --filter_scaling 1 2\
  --kernel_size 3\
  --layers 2 3\
  --block_conv_layers 2\
  --input_kernel_size 3 5 7\
  --activation leaky_relu tanh\
  --epochs 200\
  --initial_learning_rate 1e-4\
  --residual_weights uniform sqrt\
  --logdir=$EXORIM_PATH/logs/\
  --model_dir=$EXORIM_PATH/models/\
  --logname_prefixe=RIMB_wide_search2\
  --seed 42
