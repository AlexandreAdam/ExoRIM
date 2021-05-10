from numpy.random import choice as choose
from exorim import RIM, MSE, PhysicalModel
from exorim.definitions import DTYPE
from exorim.interferometry.simulated_data import CenteredBinaries
from argparse import ArgumentParser
from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd



# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))


AUTOTUNE = tf.data.experimental.AUTOTUNE


def hparams_for_gridsearchV2(n_params):
    for i in range(n_params):
        yield {
                # "grad_log_scale": choose([False]),
                # "logim": choose([True]),
                "time_steps": choose([6, 8, 10, 12]),
                "state_depth": choose([16, 32, 64, 128, 256]),
                # "kernel_size_downsampling": choose([[3, 3], [5, 5], [7, 7]]), #choose([3, 5, 7])
                "filters_downsampling": choose([16, 32, 64]),
                "downsampling_layers": choose([1, 2]),
                "conv_layers": choose([1, 2]),
                # "kernel_size_gru":  choose([[3, 3], [5, 5], [7, 7]]),
                "hidden_layers": choose([1, 2]),
                # "kernel_size_upsampling": choose([3, 5, 7]),
                "filters_upsampling": choose([16, 32, 64]),
                "kernel_regularizer_amp": choose([1e-3, 1e-2, 1e-1]),
                "bias_regularizer_amp": choose([1e-3, 1e-2, 1e-1]),
                "batch_norm": choose([True, False]),
                "activation": choose(["leaky_relu", "relu", "gelu", "elu"]),
                "initial_learning_rate": choose([1e-2, 1e-3, 1e-4]),
                "beta_1": np.round(np.random.uniform(0.5, 0.99),2),  # Adam update in RIM
                "beta_2": np.round(10**(-(np.random.uniform(0, 0.1))), 3),
                # "batch_size": choose([5, 10, 20]),
                # "temperature": choose([1, 10, 100]),
            }


parser = ArgumentParser()
# note that number of fits is model_trained * folds, default is then 50 fits done
parser.add_argument("--model_trained", type=int, default=10)
# parser.add_argument("-f", "--folds", type=int, default=5)
parser.add_argument("--decay_rate", type=float, default=0.9)
parser.add_argument("--decay_steps", type=int, default=1000)
parser.add_argument("-n", "--number_images", type=int, default=500)
parser.add_argument("-s", "--split", type=float, default=0.8)
parser.add_argument("-b", "--batch_size", type=int, default=25, help="Batch size")
parser.add_argument("-t", "--training_time", type=float, default=1, help="Time allowed for training in hours")
parser.add_argument("-m", "--min_delta", type=float, default=0, help="Tolerance for early stopping")
parser.add_argument("-p", "--patience", type=int, default=10, help="Patience for early stopping")
# parser.add_argument("-c", "--checkpoint", type=int, default=20, help="Checkpoint to save model weights")
parser.add_argument("-e", "--max_epoch", type=int, default=100, help="Maximum number of epoch")
# parser.add_argument("--index_save_mod", type=int, default=25, help="Image index to be saved")
# parser.add_argument("--epoch_save_mod", type=int, default=5, help="Epoch at which to save images")
# step save mod is overwritten by hparams

args = parser.parse_args()
date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

PARAM_GRID = hparams_for_gridsearchV2(args.model_trained)
SCOREFILE = os.path.expanduser('./scores_1.csv')

phys = PhysicalModel(pixels=64)

meta_data = CenteredBinaries(
    total_items=args.number_images,
    pixels=64,
    width=6,
    flux=64**2
)
images = tf.constant(meta_data.generate_epoch_images(), dtype=DTYPE)
noisy_data = phys.noisy_forward(images)
X = tf.data.Dataset.from_tensor_slices(noisy_data)  # split along batch dimension
Y = tf.data.Dataset.from_tensor_slices(images)
dataset = tf.data.Dataset.zip((X, Y))
dataset = dataset.batch(args.batch_size, drop_remainder=True)
dataset = dataset.cache()  # accelerate the second and subsequent iterations over the dataset
dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs

metrics = {
    "chi_squared": lambda Y_pred, Y_true: tf.reduce_mean(phys.chi_squared(Y_pred, phys.forward(Y_true)))
}

cost_function = MSE()


# performs hyperparamter search in parallel, using a slurm array job
def search_distributed():
    for i in range(this_worker, args.model_trained, N_WORKERS):
        params = next(PARAM_GRID)
        exclude_keys = ['initial_learning_rate']
        rim_params = {k: params[k] for k in set(list(params.keys())) - set(exclude_keys)}
        rim = RIM(phys, **rim_params)
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params["initial_learning_rate"],
            decay_rate=args.decay_rate,
            decay_steps=args.decay_steps
        )
        history = rim.fit(
            train_dataset=dataset,
            # test_dataset=test_dataset,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
            max_time=args.training_time,
            cost_function=cost_function,
            min_delta=args.min_delta,
            patience=args.patience,
            # checkpoints=args.checkpoint,
            # output_dir=fold_dir,
            # checkpoint_dir=models_dir,
            max_epochs=args.max_epoch,
            # output_save_mod={"index_mod": args.index_save_mod,
            #                  "epoch_mod": args.epoch_save_mod,
            #                  "step_mod": hparams["steps"]}, # save first and last step images
            metrics=metrics,
            # name=f"rim_{hparams['grid_id']:03}_{fold:02}"
        )
        score = np.min(history["train_loss"])
        chi_squared = np.min(history["chi_squared_train"])
        out = params
        out.update({"train_loss": score, "chi_squared_train": chi_squared})
        result = pd.DataFrame.from_dict(out, orient="index").T
        if i == 0:
            result.to_csv(SCOREFILE, mode="w", header=True)
        else:
            result.to_csv(SCOREFILE, mode="a", header=False)


def main():
    print(f'WORKER {this_worker} ALIVE.')
    search_distributed()
    print(f'WORKER {this_worker} DONE.')


if __name__ == '__main__':
    main()
