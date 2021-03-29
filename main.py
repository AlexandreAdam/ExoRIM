from exorim import RIM, MSE, PhysicalModel
from preprocessing.simulate_data import create_and_save_data
from exorim.interferometry.simulated_data import CenteredBinaries 
from exorim.definitions import DTYPE
from argparse import ArgumentParser
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
# try:
#     import wandb
#     wandb.init(project="exorim", sync_tensorboard=True)
# except ImportError:
#     print("wandb not installed, package ignored")

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_datasets(meta_data, rim, dirname, batch_size=None, index_save_mod=1, format="txt"):
    images = tf.convert_to_tensor(create_and_save_data(dirname, meta_data, index_save_mod, format), dtype=DTYPE)
    noisy_data = rim.physical_model.forward(images)     # TODO make this noisy forward
    X = tf.data.Dataset.from_tensor_slices(noisy_data)  # split along batch dimension
    Y = tf.data.Dataset.from_tensor_slices(images)
    dataset = tf.data.Dataset.zip((X, Y))
    if batch_size is not None:  # for train set
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.cache()             # accelerate the second and subsequent iterations over the dataset
        dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs
    else:
        # batch together all examples, for test set
        dataset = dataset.batch(images.shape[0], drop_remainder=True)
        dataset = dataset.cache()
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pixels", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--decay_steps", type=int, default=10)
    parser.add_argument("-n", "--number_images", type=int, default=100)
    parser.add_argument("-w", "--wavelength", type=float, default=0.5e-6)
    parser.add_argument("--SNR", type=float, default=10, help="Signal to noise ratio")
    parser.add_argument("-s", "--split", type=float, default=0.8)
    parser.add_argument("-b", "--batch", type=int, default=10, help="Batch size")
    # parser.add_argument("-t", "--training_time", type=float, default=2, help="Time allowed for training in hours")
    parser.add_argument("-m", "--min_delta", type=float, default=0., help="Tolerance for early stopping")
    parser.add_argument("-p", "--patience", type=int, default=10, help="Patience for early stopping") # infinite patience for hparam check
    parser.add_argument("-c", "--checkpoint", type=int, default=5, help="Checkpoint to save model weights")
    parser.add_argument("-e", "--max_epoch", type=int, default=50, help="Maximum number of epoch")
    parser.add_argument("--index_save_mod", type=int, default=20, help="Image index to be saved")
    parser.add_argument("--epoch_save_mod", type=int, default=1, help="Epoch at which to save images")
    parser.add_argument("--noise_floor", type=float, default=1, help="Intensity noise floor")
    parser.add_argument("--format", type=str, default="png", help="Format with which to save image, either png or txt")
    parser.add_argument("--seed", type=float, default=42, help="Dataset seed")

    args = parser.parse_args()
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    id = date
    print(f"id = {id}")

    basedir = os.getcwd()  # assumes script is run from base directory
    results_dir = os.path.join(basedir, "results", id)
    os.mkdir(results_dir)
    models_dir = os.path.join(basedir, "models", id)
    os.mkdir(models_dir)
    data_dir = os.path.join(basedir, "data", id)
    os.mkdir(data_dir)
    train_dir = os.path.join(data_dir, "train")
    os.mkdir(train_dir)
    test_dir = os.path.join(data_dir, "test")
    os.mkdir(test_dir)

    # another approach to save results using tensorboard and wandb
    logdir = os.path.join(basedir, "logs", id)
    os.mkdir(logdir)
    os.mkdir(os.path.join(logdir, "train"))
    os.mkdir(os.path.join(logdir, "test"))

    phys = PhysicalModel( # GOLAY9 mask
        pixels=args.pixels,
        wavelength=args.wavelength,
        SNR=args.SNR
    )
    metrics = {
        "Chi squared": lambda Y_pred, Y_true: tf.reduce_mean(phys.chi_squared(Y_pred, phys.forward(Y_true)))
    }

    rim = RIM(physical_model=phys, noise_floor=args.noise_floor)
    train_meta = CenteredBinaries(total_items=int(args.split * args.number_images), pixels=args.pixels, width=3, seed=args.seed)
    testmeta = CenteredBinaries(total_items=int((1 - args.split) * args.number_images), pixels=args.pixels, width=3, seed=0)

    train_dataset = create_datasets(train_meta, rim, train_dir, batch_size=args.batch, index_save_mod=args.index_save_mod, format="txt")
    test_dataset = create_datasets(testmeta, rim, test_dir, batch_size=None, index_save_mod=1, format="txt")
    cost_function = MSE()
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps
    )
    history = rim.fit(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
        # max_time=args.training_time,
        cost_function=cost_function,
        min_delta=args.min_delta,
        patience=args.patience,
        metrics=metrics,
        track="train_loss",
        checkpoints=args.checkpoint,
        output_dir=results_dir,
        checkpoint_dir=models_dir,
        max_epochs=args.max_epoch,
        logdir=logdir,
        output_save_mod={
            "index_mod": args.index_save_mod,
            "epoch_mod": args.epoch_save_mod,
            "step_mod": 5,
            "timestep_mod": 1
        }
    )
    for key, item in history.items():
        np.savetxt(os.path.join(results_dir, key + ".txt"), item)
