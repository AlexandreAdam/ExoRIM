from exorim import RIM, PhysicalModel
from exorim.definitions import DTYPE, RIM_HPARAMS, MODEL_W_INVERSE_HPARAMS
from exorim.datasets import NCompanions
from exorim.models import Model, UnetModel, UnetModelwithInverseFunc
from exorim.utils import residual_plot, plot_to_image
from datetime import datetime
import os, time, json
import tensorflow as tf
import numpy as np
from exorim.utils import nullwriter


def main(args):
    print(args)
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    if args.json_override is not None:
        if isinstance(args.json_override, list):
            files = args.json_override
        else:
            files = [args.json_override, ]
        for file in files:
            with open(file, "r") as f:
                json_override = json.load(f)
            args_dict = vars(args)
            args_dict.update(json_override)

    phys = PhysicalModel(
        pixels=args.pixels,
        wavelength=args.wavelength,
        logim=True,
        oversampling_factor=args.oversampling_factor,
        chi_squared=args.chi_squared,
        plate_scale=args.plate_scale
    )

    train_dataset = NCompanions(
        phys=phys,
        total_items=args.total_items,
        batch_size=args.batch_size,
        width=args.width
    )
    model = UnetModelwithInverseFunc(
        pixels=phys.pixels,
        number_of_baselines=phys.nbuv,
        number_of_closure_phases=phys.nbcp,
        filters=args.filters,
        kernel_size=args.kernel_size,
        filter_scaling=args.filter_scaling,
        input_kernel_size=args.input_kernel_size,
        layers=args.layers,
        block_conv_layers=args.block_conv_layers,
        strides=args.strides,
        activation=args.activation,
        upsampling_interpolation=args.upsampling_interpolation,
        inverse_layers=args.inverse_layers,
        inverse_filters=args.inverse_filters
    )

    vars(args)["inverse_function"] = "model"
    rim = RIM(
        model=model,
        physical_model=phys,
        steps=args.steps,
        inverse_function=args.inverse_function
    )

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.initial_learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        staircase=args.staircase
    )
    optim = tf.keras.optimizers.deserialize(
        {
            "class_name": args.optimizer,
            'config': {"learning_rate": learning_rate_schedule}
        }
    )
    # weights for time steps in the loss function
    if args.time_weights == "uniform":
        wt = tf.ones(shape=(args.steps), dtype=DTYPE) / args.steps
    elif args.time_weights == "linear":
        wt = 2 * (tf.range(args.steps, dtype=DTYPE) + 1) / args.steps / (args.steps + 1)
    elif args.time_weights == "quadratic":
        wt = 6 * (tf.range(args.steps, dtype=DTYPE) + 1) ** 2 / args.steps / (args.steps + 1) / (2 * args.steps + 1)
    else:
        raise ValueError("time_weights must be in ['uniform', 'linear', 'quadratic']")
    wt = wt[..., tf.newaxis]  # [steps, batch]

    if args.residual_weights == "uniform":
        w = tf.keras.layers.Lambda(lambda s: tf.ones_like(s, dtype=DTYPE) / tf.cast(tf.math.reduce_prod(s.shape[1:]), DTYPE))
    elif args.residual_weights == "linear":
        w = tf.keras.layers.Lambda(lambda s: s / tf.reduce_sum(s, axis=(1, 2, 3), keepdims=True))
    elif args.residual_weights == "quadratic":
        w = tf.keras.layers.Lambda(lambda s: tf.square(s) / tf.reduce_sum(tf.square(s), axis=(1, 2, 3), keepdims=True))
    elif args.residual_weights == "sqrt":
        w = tf.keras.layers.Lambda(lambda s: tf.sqrt(s) / tf.reduce_sum(tf.sqrt(s), axis=(1, 2, 3), keepdims=True))
    else:
        raise ValueError("residual_weights must be in ['uniform', 'linear', 'quadratic', 'sqrt']")

    # ==== Take care of where to write logs and stuff =================================================================
    if args.model_id.lower() != "none":
        if args.logname is not None:
            logname = args.model_id + "_" + args.logname
            model_id = args.model_id
        else:
            logname = args.model_id + "_" + datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            model_id = args.model_id
    elif args.logname is not None:
        logname = args.logname
        model_id = logname
    else:
        logname = args.logname_prefixe + "_" + datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        model_id = logname
    if args.logdir.lower() != "none":
        logdir = os.path.join(args.logdir, logname)
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        writer = tf.summary.create_file_writer(logdir)
    else:
        writer = nullwriter()
    # ===== Make sure directory and checkpoint manager are created to save model ===================================
    if args.model_dir.lower() != "none":
        checkpoints_dir = os.path.join(args.model_dir, logname)
        old_checkpoints_dir = os.path.join(args.model_dir, model_id)  # in case they differ we load model from a different directory
        if not os.path.isdir(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            with open(os.path.join(checkpoints_dir, "script_params.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
            with open(os.path.join(checkpoints_dir, "model_hparams.json"), "w") as f:
                hparams_dict = {key: vars(args)[key] for key in MODEL_W_INVERSE_HPARAMS}
                json.dump(hparams_dict, f, indent=4)
            with open(os.path.join(checkpoints_dir, "rim_hparams.json"), "w") as f:
                hparams_dict = {key: vars(args)[key] for key in RIM_HPARAMS}
                json.dump(hparams_dict, f, indent=4)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=rim.model)
        checkpoint_manager = tf.train.CheckpointManager(ckpt, old_checkpoints_dir, max_to_keep=args.max_to_keep)
        save_checkpoint = True
        # ======= Load model if model_id is provided ===============================================================
        if args.model_id.lower() != "none":
            checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint)
        if old_checkpoints_dir != checkpoints_dir:  # save progress in another directory.
            if args.reset_optimizer_states:
                optim = tf.keras.optimizers.deserialize(
                    {
                        "class_name": args.optimizer,
                        'config': {"learning_rate": learning_rate_schedule}
                    }
                )
                ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optim, net=rim.model)
            checkpoint_manager = tf.train.CheckpointManager(ckpt, checkpoints_dir, max_to_keep=args.max_to_keep)
    else:
        save_checkpoint = False

    # =================================================================================================================

    def train_step(X, Y, noise_rms):
        with tf.GradientTape() as tape:
            tape.watch(rim.model.trainable_variables)
            y_pred_series, chi_squared = rim.call(X, noise_rms, outer_tape=tape)
            # weighted mean over image residuals
            cost = tf.reduce_sum(w(Y) * tf.square(y_pred_series - rim.inverse_link_function(Y)), axis=(2, 3, 4))
            # weighted mean over time steps
            cost = tf.reduce_sum(wt * cost, axis=0)
            # final cost is mean over global batch size
            cost = tf.reduce_mean(cost)
        gradient = tape.gradient(cost, rim.model.trainable_variables)
        gradient = [tf.clip_by_norm(grad, 5.) for grad in gradient]
        optim.apply_gradients(zip(gradient, rim.model.trainable_variables))
        # Update metrics with "converged" score
        chi_squared = tf.reduce_mean(chi_squared[-1])
        return cost, chi_squared

    # ====== Training loop ============================================================================================
    time_per_step = tf.metrics.Mean()
    epoch_loss = tf.metrics.Mean()
    epoch_chi_squared = tf.metrics.Mean()
    history = {  # recorded at the end of an epoch only
        "train_chi_squared": [],
        "train_cost": [],
        "learning_rate": [],
        "time_per_step": [],
        "step": [],
        "wall_time": []
    }
    best_loss = np.inf
    patience = args.patience
    step = 0
    global_start = time.time()
    estimated_time_for_epoch = 0
    out_of_time = False
    lastest_checkpoint = 1
    for epoch in range(args.epochs):
        if (time.time() - global_start) > args.max_time * 3600 - estimated_time_for_epoch:
            break
        epoch_start = time.time()
        epoch_loss.reset_states()
        epoch_chi_squared.reset_states()
        time_per_step.reset_states()
        with writer.as_default():
            for batch, (X, Y, noise_rms) in enumerate(train_dataset):
                start = time.time()
                cost, chi_squared = train_step(X, Y, noise_rms)
                # ========== Summary and logs =========================================================================
                _time = time.time() - start
                time_per_step.update_state([_time])
                epoch_loss.update_state([cost])
                epoch_chi_squared.update_state([chi_squared])
                step += 1
            # last batch we make a summary of residuals
            if args.n_residuals > 0:
                tf.summary.image(f"Residuals",
                                 plot_to_image(
                                     residual_plot(
                                         train_dataset,
                                         rim,
                                         args.n_residuals
                                     )), step=step)

            train_cost = epoch_loss.result().numpy()
            train_chi_sq = epoch_chi_squared.result().numpy()
            tf.summary.scalar("Time per step", time_per_step.result(), step=step)
            tf.summary.scalar("Chi Squared", train_chi_sq, step=step)
            tf.summary.scalar("MSE", train_cost, step=step)
            tf.summary.scalar("Learning Rate", optim.lr(step), step=step)
        print(f"epoch {epoch} | train loss {train_cost:.3e} "
              f"| lr {optim.lr(step).numpy():.2e} | time per step {time_per_step.result().numpy():.2e} s"
              f"| chi sq {train_chi_sq:.2e}")
        history["train_cost"].append(train_cost)
        history["learning_rate"].append(optim.lr(step).numpy())
        history["train_chi_squared"].append(train_chi_sq)
        history["time_per_step"].append(time_per_step.result().numpy())
        history["step"].append(step)
        history["wall_time"].append(time.time() - global_start)

        cost = train_cost
        if np.isnan(cost):
            print("Training broke the Universe")
            break
        if cost < (1 - args.tolerance) * best_loss:
            best_loss = cost
            patience = args.patience
        else:
            patience -= 1
        if (time.time() - global_start) > args.max_time * 3600:
            out_of_time = True
        if save_checkpoint:
            checkpoint_manager.checkpoint.step.assign_add(1)  # a bit of a hack
            if epoch % args.checkpoints == 0 or patience == 0 or epoch == args.epochs - 1 or out_of_time:
                with open(os.path.join(checkpoints_dir, "score_sheet.txt"), mode="a") as f:
                    np.savetxt(f, np.array([[lastest_checkpoint, cost]]))
                lastest_checkpoint += 1
                checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint_manager.checkpoint.step), checkpoint_manager.latest_checkpoint))
        if patience == 0:
            print("Reached patience")
            break
        if out_of_time:
            break
        if epoch > 0:
            estimated_time_for_epoch = time.time() - epoch_start
        if optim.lr(step).numpy() < 1e-8:
            print("Reached learning rate limit")
            break
    print(f"Finished training after {(time.time() - global_start) / 3600:.3f} hours.")
    return history, best_loss


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_id",           default="None",                 help="Start from this model id checkpoint. None means start from scratch")
    parser.add_argument("--architecture",       default="unet",                 help="Argument must be one of ['hourglass', 'unet', 'unet_with_inverse']")

    # Binary dataset parameters
    parser.add_argument("--total_items",        default=10,     type=int,       help="Total items in an epoch")
    parser.add_argument("--batch_size",         default=1,      type=int)
    parser.add_argument("--width",              default=2,      type=float,     help="Sigma parameter of super-gaussian in pixel units")

    # Physical Model parameters
    parser.add_argument("--wavelength",         default=3.8e-6,     type=float,     help="Wavelength in meters")
    parser.add_argument("--oversampling_factor", default=None,         type=float,  help="Set the pixels size = resolution / oversampling_factor. Resolution is set by Michelson criteria")
    parser.add_argument("--chi_squared",        default="append_visibility_amplitude_closure_phase",    help="One of 'visibility' or 'append_visibility_amplitude_closure_phase'. Default is the latter.")
    parser.add_argument("--pixels",             default=128,        type=int)
    parser.add_argument("--redundant",          action="store_true",                help="Whether to use redundant closure phase in likelihood or not")
    parser.add_argument("--plate_scale",        default=5,          type=float,     help="Size of a pixel, in mas")
    parser.add_argument("--beta",               default=1,          type=float,     help="Lagrange multiplier for the closure phase term.")

    # RIM hyper parameters
    parser.add_argument("--steps",              default=6,          type=int,       help="Number of recurrent steps in the model")

    # Neural network hyper parameters
    parser.add_argument("--filters",                                    default=32,     type=int)
    parser.add_argument("--filter_scaling",                             default=2,      type=float)
    parser.add_argument("--kernel_size",                                default=3,      type=int)
    parser.add_argument("--layers",                                     default=2,      type=int)
    parser.add_argument("--block_conv_layers",                          default=2,      type=int)
    parser.add_argument("--strides",                                    default=2,      type=int)
    parser.add_argument("--input_kernel_size",                          default=7,      type=int)
    parser.add_argument("--upsampling_interpolation",                   action="store_true")
    parser.add_argument("--activation",                                 default="tanh")
    parser.add_argument("--initializer",                                default="glorot_normal")
    parser.add_argument("--inverse_layers",                             default=2,      type=int,   help="Number of conv layers for the inverse function in the model")
    parser.add_argument("--inverse_filters",                            default=32,     type=int,   help="Number filters for each conv layers for the inverse function in the model")

    # Optimization params
    parser.add_argument("--epochs",                 default=10,     type=int,      help="Number of epochs for training.")
    parser.add_argument("--optimizer",              default="Adamax",               help="Class name of the optimizer (e.g. 'Adam' or 'Adamax')")
    parser.add_argument("--initial_learning_rate",  default=1e-4,   type=float,     help="Initial learning rate.")
    parser.add_argument("--decay_rate",             default=1.,     type=float,     help="Exponential decay rate of learning rate (1=no decay).")
    parser.add_argument("--decay_steps",            default=1000,   type=int,       help="Decay steps of exponential decay of the learning rate.")
    parser.add_argument("--staircase",              action="store_true",            help="Learning rate schedule only change after decay steps if enabled.")
    parser.add_argument("--patience",               default=np.inf, type=int,       help="Number of step at which training is stopped if no improvement is recorder.")
    parser.add_argument("--tolerance",              default=0,      type=float,     help="Current score <= (1 - tolerance) * best score => reset patience, else reduce patience.")
    parser.add_argument("--track_train",            action="store_true",            help="Track training metric instead of validation metric, in case we want to overfit")
    parser.add_argument("--max_time",               default=np.inf, type=float,     help="Time allowed for the training, in hours.")
    parser.add_argument("--time_weights",           default="uniform",              help="uniform: w_t=1 for all t, linear: w_t~t, quadratic: w_t~t^2")
    parser.add_argument("--reset_optimizer_states",  action="store_true",           help="When training from pre-trained weights, reset states of optimizer.")
    parser.add_argument("--residual_weights",        default="sqrt",                help="Options are ['uniform', 'linear', 'quadratic', 'sqrt']")

    # logs
    parser.add_argument("--logdir",                  default="None",                help="Path of logs directory. Default if None, no logs recorded.")
    parser.add_argument("--model_dir",               default="None",                help="Path to the directory where to save models checkpoints.")
    parser.add_argument("--logname",                 default=None,                  help="Overwrite name of the log with this argument")
    parser.add_argument("--logname_prefixe",         default="RIM",                 help="If name of the log is not provided, this prefix is prepended to the date")
    parser.add_argument("--checkpoints",             default=10,    type=int,       help="Save a checkpoint of the models each {%} iteration.")
    parser.add_argument("--max_to_keep",             default=3,     type=int,       help="Max model checkpoint to keep.")
    parser.add_argument("--n_residuals",             default=2,     type=int,       help="Number of residual plots to save. Add overhead at the end of an epoch only. Should be >= 2.")

    # Reproducibility params
    parser.add_argument("--seed",                   default=42,      type=int,      help="Random seed for numpy and tensorflow.")
    parser.add_argument("--json_override",          default=None,    nargs="+",     help="A json filepath that will override every command line parameters. Useful for reproducibility")
    args = parser.parse_args()
    main(args)
