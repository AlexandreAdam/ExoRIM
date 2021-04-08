import ray
import wandb
from ray import tune
from ray.tune.integration.wandb import wandb_mixin
from exorim import MSE, PhysicalModel, RIM
from exorim.interferometry.simulated_data import CenteredBinaries
from exorim.definitions import DTYPE
from ray.tune.suggest.hyperopt import HyperOptSearch

# from ray.tune.schedulers import HyperBandScheduler

#
# class MyTrainableClass(tune.Trainable):
#     """Example agent whose learning curve is a random sigmoid.
#
#     The dummy hyperparameters "width" and "height" determine the slope and
#     maximum reward value reached.
#     """
#
#     def setup(self, config):
#         self.timestep = 0
#
#     def step(self):
#         self.timestep += 1
#         v = np.tanh(float(self.timestep) / self.config.get("width", 1))
#         v *= self.config.get("height", 1)
#         time.sleep(0.1)
#
#         # Here we use `episode_reward_mean`, but you can also report other
#         # objectives such as loss or accuracy.
#         return {"episode_reward_mean": v}
#
#     def save_checkpoint(self, checkpoint_dir):
#         path = os.path.join(checkpoint_dir, "checkpoint")
#         with open(path, "w") as f:
#             f.write(json.dumps({"timestep": self.timestep}))
#         return path
#
#     def load_checkpoint(self, checkpoint_path):
#         with open(checkpoint_path) as f:
#             self.timestep = json.loads(f.read())["timestep"]

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--ngpus", type=int, default=0)
    parser.add_argument("--pixels", type=int, default=32)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        max_epochs = 2
    else:
        max_epochs = 10 # 20
    wandb.init(project="exorim_modelv1_raytunev1")

    config = {
        "grad_log_scale": tune.choice([True, False]),
        "logim": tune.choice([True, False]),
        "lam": tune.choice([0, 1, 1e-2, 100]),
        "time_steps": tune.choice([6, 8, 10, 12]),
        "state_depth": tune.choice([16, 32, 64, 128, 256]),
        "kernel_size_downsampling": tune.choice([3, 5, 7]),
        "filters_downsampling": tune.choice([16, 32, 64]),
        "downsampling_layers": tune.choice([1, 2, 3]),
        "conv_layers": tune.choice([1, 2]),
        "kernel_size_gru": tune.choice([3, 5, 7]),
        "hidden_layers": tune.choice([1, 2]),
        "kernel_size_upsampling": tune.choice([3, 5, 7]),
        "filters_upsampling": tune.choice([16, 32, 64]),
        "kernel_regularizer_amp": tune.choice([1e-3, 1e-2, 1e-1]),
        "bias_regularizer_amp": tune.choice([1e-3, 1e-2, 1e-1]),
        "batch_norm": tune.choice([True, False]),
        "activation": tune.choice(["leaky_relu", "relu", "gelu", "elu"]),
        "initial_learning_rate": tune.choice([1e-3, 1e-4, 1e-5]),
        "beta_1": tune.uniform(0.5, 0.99), # Adam optimzier
        "batch_size": tune.choice([5, 10, 20]),
        "temperature": tune.choice([1, 10, 100, 1e4, 1e5]),
        "wandb": {
            "project": "exorim_modelv1_raytunev1",
            }
    }

    @wandb_mixin
    def trainable(config):
        import tensorflow as tf
        phys = PhysicalModel(args.pixels, temperature=config["temperature"], logim=config["logim"], lam=config["lam"])
        Y = tf.convert_to_tensor(CenteredBinaries(args.images, args.pixels, width=3, flux=args.pixels**2, seed=42).generate_epoch_images(), dtype=DTYPE)
        X = phys.forward(Y)
        Y = tf.data.Dataset.from_tensor_slices(Y)
        X = tf.data.Dataset.from_tensor_slices(X)
        dataset = tf.data.Dataset.zip((X, Y))
        dataset = dataset.batch(config["batch_size"], drop_remainder=True)
        # dataset = dataset.cache()
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        rim = RIM(
            phys,
            time_steps=config["time_steps"],
            state_depth=config["state_depth"],
            grad_log_scale=config["grad_log_scale"],
            kernel_size_downsampling=config["kernel_size_downsampling"],
            filters_downsampling=config["filters_downsampling"],
            downsampling_layers=config["downsampling_layers"],
            conv_layers=config["conv_layers"],
            kernel_size_gru=config["kernel_size_gru"],
            hidden_layers=config["hidden_layers"],
            filters_upsampling=config["filters_upsampling"],
            activation=config["activation"],
            kernel_regularizer_amp=config["kernel_regularizer_amp"],
            bias_regularizer_amp=config["bias_regularizer_amp"]
                  )
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["initial_learning_rate"],
            decay_rate=0.9,
            decay_steps=10
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler, beta_1=config["beta_1"])
        history = rim.fit(dataset, MSE(), optimizer, max_epochs=max_epochs)
        tune.report(train_loss = min(history["train_loss"]))
        wandb.log({"train_loss": min(history["train_loss"])})

    search_algorithm = HyperOptSearch(metric="train_loss", mode="min")
    analysis = tune.run(
        trainable,
        config=config,
        num_samples=5,
        search_alg=search_algorithm,
        resources_per_trial={"cpu": 2}
        )

    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    ray.shutdown()
