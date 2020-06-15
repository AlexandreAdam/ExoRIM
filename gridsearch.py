from ExoRIM.gridsearch import hparams_for_gridsearchV1, kfold_splits
from ExoRIM.model import RIM, CostFunction
from ExoRIM.simulated_data import CenteredImagesv1, OffCenteredBinaries
from preprocessing.simulate_data import create_and_save_data
from argparse import ArgumentParser
from datetime import datetime
import pickle
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
    parser.add_argument("-c", "--checkpoint", type=int, default=20, help="Checkpoint to save model weights")
    parser.add_argument("-e", "--max_epoch", type=int, default=20, help="Maximum number of epoch")
    parser.add_argument("--index_save_mod", type=int, default=25, help="Image index to be saved")
    parser.add_argument("--epoch_save_mod", type=int, default=5, help="Epoch at which to save images")
    # step save mod is overwritten by hparams

    args = parser.parse_args()
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    basedir = os.getcwd()  # assumes script is run from base directory
    projector_dir = os.path.join(basedir, "data", "projector_arrays")
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

    # metrics only support grey scale images
    metrics = {
        "ssim": lambda Y_pred, Y_true: tf.image.ssim(Y_pred, Y_true, max_val=1.0),
        # Bug is tf 2.0.0, make sure filter size is small enough such that H/2**4 and W/2**4 >= filter size
        # alternatively (since H/2**4 is = 1 in our case), it is possible to lower the power factors such that
        # H/(2**(len(power factor)-1)) > filter size
        # Hence, using 3 power factors with filter size=2 works, and so does 2 power factors with filter_size <= 8
        # paper power factors are [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        # After test, it seems filter_size=11 also works with 2 power factors and 32 pixel image
        "ssim_multiscale_01": lambda Y_pred, Y_true: tf.image.ssim_multiscale(Y_pred, Y_true, max_val=1.0,
                                                                              filter_size=11,
                                                                              power_factors=[0.0448, 0.2856]),
        "ssim_multiscale_23": lambda Y_pred, Y_true: tf.image.ssim_multiscale(Y_pred, Y_true, max_val=1.0,
                                                                              filter_size=11,
                                                                              power_factors=[0.3001, 0.2363]),
        "ssim_multiscale_34": lambda Y_pred, Y_true: tf.image.ssim_multiscale(Y_pred, Y_true, max_val=1.0,
                                                                              filter_size=11,
                                                                              power_factors=[0.2363, 0.1333])
    }
    # meta_data = OffCenteredBinaries(
    #     total_items=args.number_images,
    #     pixels=32
    # )

    Y = tf.convert_to_tensor(create_and_save_data(data_dir, meta_data), dtype=tf.float32)

    for hparams in hparams_for_gridsearchV1(args.model_trained):
        hparams_dir = os.path.join(results_dir, f"hparams_{hparams['grid_id']:03}")
        os.mkdir(hparams_dir)
        hparams["batch"] = int(args.batch)
        hparams["date"] = date
        holes = hparams["mask_holes"]
        # make sure projectors were precomputed
        mask_coordinates = np.loadtxt(os.path.join(projector_dir, f"mask_{holes}_holes.txt"))
        with open(os.path.join(projector_dir, f"projectors_{holes}_holes.pickle"), "rb") as f:
            arrays = pickle.load(f)

        rim = RIM(mask_coordinates=mask_coordinates, hyperparameters=hparams, arrays=arrays)
        X = rim.physical_model.simulate_noisy_image(Y)
        with open(os.path.join(models_dir, f"hyperparameters_{hparams['grid_id']:03}.json"), "w") as f:
            json.dump(hparams, f)

        cost_function = CostFunction()

        # multiprocessing for this would be hard with a GPU since processes would hang up when calling GPU
        # Tensorflow, by default, allocate all the memory of the GPU for the task at hand.

        fold = 0
        for train_dataset, test_dataset in kfold_splits(X, Y, args.folds, args.batch):
            fold_dir = os.path.join(hparams_dir, f"fold_{fold:02}")
            os.mkdir(fold_dir)
            # reset the model for each fold
            rim = RIM(mask_coordinates=mask_coordinates, hyperparameters=hparams, arrays=arrays)
            history = rim.fit(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                max_time=args.training_time,
                cost_function=cost_function,
                min_delta=args.min_delta,
                patience=args.patience,
                checkpoints=args.checkpoint,
                output_dir=fold_dir,
                checkpoint_dir=models_dir,
                max_epochs=args.max_epoch,
                output_save_mod={"index_mod": args.index_save_mod,
                                 "epoch_mod": args.epoch_save_mod,
                                 "step_mod": hparams["steps"]}, # save first and last step imagees
                metrics=metrics,
                name=f"rim_{hparams['grid_id']:03}_{fold:02}"
            )
            for key, item in history.items():
                np.savetxt(os.path.join(fold_dir, key + ".txt"), item)
            fold += 1


