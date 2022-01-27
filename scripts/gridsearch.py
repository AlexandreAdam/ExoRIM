from numpy.random import choice
from exorim import RIM, PhysicalModel
from exorim.definitions import DTYPE, AUTOTUNE
from exorim.simulated_data import CenteredBinariesDataset
from exorim.models import Modelv2
from argparse import ArgumentParser
from datetime import datetime
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


def choose_hparams():
        return {
                "logim": choice([True, False]),
                "time_steps": choice([6, 8, 10, 12]),
                "filters": choice([16, 32, 64, 128]),
                "conv_layers": choice([1, 2, 3]),
                "layers": choice([1, 2, 3]),
                "activation": choice(["leaky_relu", "relu", "gelu", "elu", "bipolar_relu"]),
                "initial_learning_rate": choice([1e-2, 1e-3, 1e-4]),
            }


def main(args):
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    hparams = choose_hparams()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    main(args)
