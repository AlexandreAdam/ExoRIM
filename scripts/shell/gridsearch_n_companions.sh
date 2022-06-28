#!/bin/bash
#SBATCH --array=1-100
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-24:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Gridsearch_exorim
#SBATCH --output=%x-%j.out
source $HOME/environments/exorim3.8/bin/activate
python $EXORIM_PATH/scripts/gridsearch.py\
  --dataset n_companions\
  --architecture unet hourglass\
  --n_models=100\
  --max_time=23.5\
  --strategy=uniform\
  --total_items 10000\
  --epochs=1000\
  --batch_size 1 5 10\
  --steps 4 8 12\
  --filters 16 32 64 128\
  --filter_scaling 1 2\
  --kernel_size 3\
  --layers 2 3 4 5\
  --block_conv_layers 1 2 3\
  --input_kernel_size 3 5 7\
  --activation leaky_relu tanh\
  --epochs 200\
  --initial_learning_rate 1e-4 1e-5\
  --decay_rate 1 0.9\
  --decay_steps 50000 100000\
  --residual_weights uniform sqrt linear\
  --logdir=$EXORIM_PATH/logs/\
  --model_dir=$EXORIM_PATH/models/\
  --logname_prefixe=RIM_wide_search\
  --seed 42
