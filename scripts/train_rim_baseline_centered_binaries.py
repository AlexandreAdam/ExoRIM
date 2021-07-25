from exorim import RIM, MSE, PhysicalModel
from exorim.simulated_data import CenteredBinaries, CenteredBinariesDataset
from exorim.physical_model import GOLAY9, JWST_NIRISS_MASK
from argparse import ArgumentParser
from datetime import datetime
import tensorflow as tf
import numpy as np
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


def main(args):
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    basedir = os.getenv("EXORIM_PATH")

    if args.model_id != "None":
        logname = args.model_id
    elif args.logname is not None:
        logname = args.logname
    else:
        logname = args.logname_prefixe + date
    results_dir = os.path.join(basedir, "results", logname)
    models_dir = os.path.join(basedir, "models", logname)
    logdir = os.path.join(basedir, "logs", logname)
    for dir in [results_dir, models_dir, logdir]:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    print(f"logname = {logname}")
    mask_coords = {
        "golay9": GOLAY9,
        "niriss": JWST_NIRISS_MASK,
        "random12": np.random.normal(size=(12, 2)),
        "random24": np.random.normal(size=(24, 2))
    }[args.mask]

    phys = PhysicalModel(
        mask_coordinates=mask_coords,
        pixels=args.pixels,
        wavelength=args.wavelength,
        SNR=args.SNR
    )

    dataset = CenteredBinariesDataset(phys, total_items=args.total_items, batch_size=args.batch_size)
    cost_function = MSE()
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps
    )

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=rim.model)
    manager = tf.train.CheckpointManager(ckpt, models_dir, max_to_keep=3)


if __name__ == "__main__":
    parser = ArgumentParser()
    # if starting from previous model version, make sure pixels match model architecture
    parser.add_argument("--model_id", type=str, default="None", help="Start from this model id checkpoint. None means start from scratch")

    # model hyperparameter
    parser.add_argument("--learning_rate",      default=1e-3,   type=float)
    parser.add_argument("--decay_rate",         default=0.9,    type=float)
    parser.add_argument("--decay_steps",        default=1000,   type=int)

    # Physics
    parser.add_argument("--wavelength",         default=0.5e-6, type=float)
    parser.add_argument("--SNR",                default=10,     type=float, help="Signal to noise ratio")
    parser.add_argument("--noise_floor",        default=1,      type=float, help="Intensity noise floor")
    parser.add_argument("--mask",               default="golay9",           help="One of golay9, niriss, random12 or random 24")

    # Training parameters
    parser.add_argument("--total_items",        default=100,    type=int)
    parser.add_argument("--train_split",        default=0.8,    type=float)
    parser.add_argument("--batch_size",         default=10,     type=int,   help="Batch size")
    parser.add_argument("--max_time",           default=2,      type=float, help="Time allowed for training in hours")
    parser.add_argument("--tolerance",          default=0.,     type=float, help="Tolerance for early stopping")
    parser.add_argument("--patience",           default=np.inf, type=int,   help="Patience for early stopping")
    parser.add_argument("--checkpoint",         default=5,      type=int,   help="Checkpoint to save model weights")
    parser.add_argument("--max_epoch",          default=100,    type=int,   help="Maximum number of epoch")
    parser.add_argument("--seed",               default=None,   type=float, help="Random seed for numpy and tensorflow")

    # logs
    parser.add_argument("--logname_prefixe",      default="RIM",              help="Name of the model")
    parser.add_argument("--logname",              default=None,               help="Overwrite the logname entirely")

    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    main(args)