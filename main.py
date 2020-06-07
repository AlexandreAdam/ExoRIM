from ExoRIM.model import RIM, CostFunction
from ExoRIM.simulated_data import CenteredImagesv1
from preprocessing.simulate_data import create_and_save_data
from argparse import ArgumentParser
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
import tarfile
import os, glob


AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_datasets(meta_data, rim, dirname, batch_size=None):
    images = tf.convert_to_tensor(create_and_save_data(dirname, meta_data), dtype=tf.float32)
    k_images = rim.physical_model.simulate_noisy_image(images)
    X = tf.data.Dataset.from_tensors(k_images)
    Y = tf.data.Dataset.from_tensors(images)
    dataset = tf.data.Dataset.zip((X, Y))
    dataset = dataset.cache()  # accelerate the second and subsequent iterations over the dataset
    if batch_size is not None:
        dataset = dataset.enumerate(start=0)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-n", "--number_images", type=int, default=10)
    parser.add_argument("-s", "--split", type=float, default=0.8)
    parser.add_argument("-b", "--batch", type=int, default=16, help="Batch size")
    parser.add_argument("-t", "--training_time", type=float, default=2, help="Time allowed for training in hours")
    parser.add_argument("--holes", type=int, default=6, help="Number of holes in the mask")
    parser.add_argument("-m", "--min_delta", type=float, default=0, help="Tolerance for early stopping")
    parser.add_argument("-p", "--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("-c", "--checkpoint", type=int, default=5, help="Checkpoint to save model weights")
    parser.add_argument("-e", "--max_epoch", type=int, default=100, help="Maximum number of epoch")
    args = parser.parse_args()
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    with open("hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)
    train_meta = CenteredImagesv1(
        total_items=args.number_images,
        pixels=hyperparameters["pixels"]
    )
    test_meta = CenteredImagesv1(
        total_items=int(args.number_images * (1 - args.split)),
        pixels=hyperparameters["pixels"]
    )
    hyperparameters["batch"] = args.batch
    hyperparameters["date"] = date
    mask_coordinates = np.random.normal(size=(args.holes, 2))
    rim = RIM(mask_coordinates=mask_coordinates, hyperparameters=hyperparameters)

    basedir = os.getcwd()  # assumes script is run from base directory
    results_dir = os.path.join(basedir, "results", date)
    os.mkdir(results_dir)
    models_dir = os.path.join(basedir, "models", date)
    os.mkdir(models_dir)
    train_dir = os.path.join(basedir, "data", date+"_train")
    os.mkdir(train_dir)
    test_dir = os.path.join(basedir, "data", date+"_test")
    os.mkdir(test_dir)

    train_dataset = create_datasets(train_meta, rim, dirname=train_dir, batch_size=args.batch)
    test_dataset = create_datasets(test_meta, rim, dirname=test_dir)
    cost_function = CostFunction()
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
        max_epochs=args.max_epoch
    )
    np.savetxt(os.path.join(results_dir, "train_loss.txt"), history["train_loss"])
    np.savetxt(os.path.join(results_dir, "test_loss.txt"), history["test_loss"])
    with open(os.path.join(models_dir, "hyperparameters.json"), "w") as f:
        json.dump(rim.hyperparameters, f)
    with tarfile.open(os.path.join(results_dir, "outputs.tar.gz"), "x:gz") as tar:
        for file in glob.glob(os.path.join(results_dir, "*.png")):
            tar.add(file)
    with tarfile.open(os.path.join(models_dir, "checkpoints.tar.gz"), "x:gz") as tar:
        for file in glob.glob(os.path.join(models_dir, "*.h5")):
            tar.add(file)
