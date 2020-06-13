from ExoRIM.gridsearch import hparams_for_gridsearchV1, leave_one_out_splits
from ExoRIM.model import RIM, CostFunction
from ExoRIM.simulated_data import CenteredImagesv1, OffCenteredBinaries
from .preprocessing.simulate_data import create_and_save_data
from argparse import ArgumentParser
from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np
import json


if __name__ == "__main__":
    parser = ArgumentParser()
    # note that number of fits is model_trained * folds, default is then 50 fits done
    parser.add_argument("--model_trained", type=int, default=10)
    parser.add_argument("-f", "--folds", type=int, default=5)
    parser.add_argument("-n", "--number_images", type=int, default=100)
    parser.add_argument("-s", "--split", type=float, default=0.8)
    parser.add_argument("-b", "--batch", type=int, default=16, help="Batch size")
    parser.add_argument("-t", "--training_time", type=float, default=0.5, help="Time allowed for training in hours")
    parser.add_argument("-m", "--min_delta", type=float, default=0, help="Tolerance for early stopping")
    parser.add_argument("-p", "--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("-e", "--max_epoch", type=int, default=100, help="Maximum number of epoch")
    parser.add_argument("--out_save_mod", type=int, default=25, help="Output index to save... The results directory "
                                                                     "can grow fast if this number is small!")
    args = parser.parse_args()
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    basedir = os.getcwd()  # assumes script is run from base directory
    results_dir = os.path.join(basedir, "results", "gridsearch_" + date)
    os.mkdir(results_dir)
    models_dir = os.path.join(basedir, "models", "gridsearch_" + date)
    os.mkdir(models_dir)
    data_dir = os.path.join(basedir, "data", "gridsearch_" + date)
    os.mkdir(data_dir)

    meta_data = CenteredImagesv1(
        total_items=args.number_images,
        pixels=32
    )
    # meta_data = OffCenteredBinaries(
    #     total_items=args.number_images,
    #     pixels=32
    # )

    images = tf.convert_to_tensor(create_and_save_data(data_dir, meta_data), dtype=tf.float32)
    mask_coordinates = np.random.normal(size=(args.holes, 2))
    np.savetxt(os.path.join(data_dir, "mask_coordinates.txt"), mask_coordinates)

    for hparams in hparams_for_gridsearchV1():
        hparams["batch"] = args.batch
        hparams["date"] = date

        rim = RIM(mask_coordinates=mask_coordinates, hyperparameters=hparams)

        k_images = rim.physical_model.simulate_noisy_image(images)
        X = tf.data.Dataset.from_tensor_slices(k_images)  # split along batch dimension
        Y = tf.data.Dataset.from_tensor_slices(images)

        cost_function = CostFunction()

        # multiprocessing for this would be hard with a GPU since processes would hang up when calling GPU
        # Tensorflow, by default, allocate all the memory of the GPU for the task at hand.

        fold = 0
        for train_dataset, test_dataset in leave_one_out_splits(X, Y, args.folds, args.batch):
            history = rim.fit(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                max_time=args.training_time,
                cost_function=cost_function,
                min_delta=args.min_delta,
                patience=args.patience,
                checkpoints=args.checkpoint,
                output_dir=results_dir,
                checkpoint_dir=models_dir,
                max_epochs=args.max_epoch,
                output_save_mod=args.out_save_mod
            )


            fold += 1
