from ExoRIM import RIM, MSE, PhysicalModel
from ExoRIM.loss import MAE, Loss
from preprocessing.simulate_data import create_and_save_data
from ExoRIM.definitions import dtype
from ExoRIM.utilities import create_dataset_from_generator, replay_dataset_from_generator
import ExoRIM.log_likelihood as chisq
from argparse import ArgumentParser
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
import os
# try:
#     import wandb
#     wandb.init(project="exorim", sync_tensorboard=True)
# except ImportError:
#     print("wandb not installed, package ignored")

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_datasets(meta_data, rim, dirname, batch_size=None, index_save_mod=1, format="png"):
    images = tf.convert_to_tensor(create_and_save_data(dirname, meta_data, index_save_mod, format), dtype=dtype)
    noisy_data = rim.physical_model.simulate_noisy_data(images)
    X = tf.data.Dataset.from_tensor_slices(noisy_data)  # split along batch dimension
    Y = tf.data.Dataset.from_tensor_slices(images)
    dataset = tf.data.Dataset.zip((X, Y))
    if batch_size is not None: # for train set
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.enumerate(start=0)
        dataset = dataset.cache()  # accelerate the second and subsequent iterations over the dataset
        dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs
    else:
        # batch together all examples, for test set
        dataset = dataset.batch(images.shape[0], drop_remainder=True)
        dataset = dataset.cache()
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hyperparameters", type=str, default="hyperparameters_vis",
                        help="Name of the hyperparameter file (without the file extension)")
    parser.add_argument("--tv", type=float, default=0., help="Total variation coefficient for the loss function")
    parser.add_argument("-n", "--number_images", type=int, default=50)
    parser.add_argument("-w", "--wavelength", type=float, default=0.5e-6)
    parser.add_argument("--SNR", type=float, default=200, help="Signal to noise ratio")
    parser.add_argument("-s", "--split", type=float, default=0.8)
    parser.add_argument("-b", "--batch", type=int, default=10, help="Batch size")
    parser.add_argument("-t", "--training_time", type=float, default=2, help="Time allowed for training in hours")
    parser.add_argument("--holes", type=int, default=21, help="Number of holes in the mask")
    parser.add_argument("--longest_baseline", type=float, default=6., help="Longest baseline (meters) in the mask, up to noise added")
    parser.add_argument("--mask_variance", type=float, default=1., help="Variance of the noise added to rho coordinate of aperture (in meter)")
    parser.add_argument("-m", "--min_delta", type=float, default=0.05, help="Tolerance for early stopping")
    parser.add_argument("-p", "--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("-c", "--checkpoint", type=int, default=5, help="Checkpoint to save model weights")
    parser.add_argument("-e", "--max_epoch", type=int, default=50, help="Maximum number of epoch")
    parser.add_argument("--index_save_mod", type=int, default=20, help="Image index to be saved")
    parser.add_argument("--epoch_save_mod", type=int, default=1, help="Epoch at which to save images")
    parser.add_argument("--scaling_factor", type=float, default=10**5, help="Maximum intensity, sets the log domain for the model input")
    parser.add_argument("--noise_floor", type=float, default=1, help="Intensity noise floor")
    parser.add_argument("--format", type=str, default="png", help="Format with which to save image, either png or txt")
    parser.add_argument("--fixed", action="store_true", help="Keeps the dataset fix for each epochs to monitor progress")
    args = parser.parse_args()
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    h_file = args.hyperparameters + ".json"
    with open(h_file, "r") as f:
        hyperparameters = json.load(f)
    try:
        name = hyperparameters["name"]
        id = date + "_" + name
    except KeyError:
        id = date

    print(f"id = {id}")
    hyperparameters["batch"] = args.batch
    hyperparameters["date"] = date
    # metrics only support grey scale images
    mae = MAE()
    metrics = {
        "ssim": lambda Y_pred, Y_true: tf.reduce_mean(tf.image.ssim(Y_pred, Y_true, max_val=1.0)),
        # Bug is tf 2.0.0, make sure filter size is small enough such that H/2**4 and W/2**4 >= filter size
        # alternatively (since H/2**4 is = 1 in our case), it is possible to lower the power factors such that
        # H/(2**(len(power factor)-1)) > filter size
        # Hence, using 3 power factors with filter size=2 works, and so does 2 power factors with filter_size <= 8
        # paper power factors are [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        # After test, it seems filter_size=11 also works with 2 power factors and 32 pixel image
        # "ssim_multiscale_01": lambda Y_pred, Y_true: tf.image.ssim_multiscale(Y_pred, Y_true, max_val=1.0,
        #                                                                       filter_size=11,
        #                                                                       power_factors=[0.0448, 0.2856]),
        # "ssim_multiscale_23": lambda Y_pred, Y_true: tf.image.ssim_multiscale(Y_pred, Y_true, max_val=1.0,
        #                                                                       filter_size=11,
        #                                                                       power_factors=[0.3001, 0.2363]),
        # "ssim_multiscale_34": lambda Y_pred, Y_true: tf.image.ssim_multiscale(Y_pred, Y_true, max_val=1.0,
        #                                                                       filter_size=11,
        #                                                                       power_factors=[0.2363, 0.1333])
        "mae": lambda Y_pred, Y_true: mae.test(Y_pred, Y_true)
    }

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

    circle_mask = np.zeros((args.holes, 2))
    for i in range(args.holes):
        circle_mask[i, 0] = (args.longest_baseline + np.random.normal(0, args.mask_variance)) * np.cos(2 * np.pi * i / args.holes)
        circle_mask[i, 1] = (args.longest_baseline + np.random.normal(0, args.mask_variance)) * np.sin(2 * np.pi * i / args.holes)
    mask = np.random.normal(0, args.longest_baseline/2, (args.holes, 2))
    phys = PhysicalModel(
        pixels=hyperparameters["pixels"],
        # mask_coordinates=circle_mask,
        mask_coordinates=mask,
        chisq_term=hyperparameters["log_likelihood_term"],
        wavelength=args.wavelength,
        SNR=args.SNR
    )
    metrics.update({
        "Chi_squared_visibilities": lambda Y_pred, Y_true: tf.reduce_mean(chisq.chi_squared_complex_visibility(
            Y_pred, phys.forward(Y_true), phys
        )),
        "Chi_squared_closure_phases": lambda Y_pred, Y_true: tf.reduce_mean(chisq.chi_squared_closure_phasor(
            Y_pred, tf.math.angle(phys.bispectrum(Y_true)), phys
        )),
        "Chi_squared_amplitude": lambda Y_pred, Y_true: tf.reduce_mean(chisq.chi_squared_amplitude(
            Y_pred, tf.math.abs(phys.forward(Y_true)), phys
        ))
    })

    rim = RIM(physical_model=phys, hyperparameters=hyperparameters, noise_floor=args.noise_floor)
    train_dataset = create_dataset_from_generator(
        physical_model=phys,
        item_per_epoch=args.number_images,
        batch_size=args.batch,
        fixed=args.fixed
    )
    test_dataset = create_dataset_from_generator(
        physical_model=phys,
        item_per_epoch=int(args.number_images * (1 - args.split)),
        batch_size=int(args.number_images * (1 - args.split)),
        fixed=args.fixed,
        seed=31415926
    )
    cost_function = Loss(tv_beta=args.tv)
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        **hyperparameters["learning rate"],
        # initial_learning_rate=1e-3,
        # decay_steps=10000,
        # decay_rate=0.90,
        staircase=True
    )
    history = rim.fit(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
        max_time=args.training_time,
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
    with open(os.path.join(models_dir, h_file), "w") as f:
        json.dump(rim.hyperparameters, f)
    # saves the ground truth images
    replay_dataset_from_generator(
        train_dataset,
        epochs=rim.hyperparameters["epoch"],
        dirname=train_dir,
        fixed=args.fixed,
        format="txt",
        index_mod=args.index_save_mod,
        epoch_mod=args.epoch_save_mod
    )
    replay_dataset_from_generator(
        test_dataset,
        epochs=rim.hyperparameters["epoch"],
        dirname=test_dir,
        fixed=args.fixed,
        format="txt",
        index_mod=int(args.number_images * (1 - args.split) / 4),
        epoch_mod=1
    )