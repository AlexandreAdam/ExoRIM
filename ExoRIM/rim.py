from ExoRIM.definitions import dtype, default_hyperparameters
from ExoRIM.utilities import save_output, save_gradient_and_weights, save_loglikelihood_grad, nullwriter
from ExoRIM.model import Model
import tensorflow as tf
import numpy as np
import time
import os


class RIM:
    def __init__(self, physical_model, hyperparameters=default_hyperparameters, dtype=dtype,
                 noise_floor=1e-16):
        self._dtype = dtype
        self.noise_floor = noise_floor
        self.hyperparameters = hyperparameters
        self.channels = hyperparameters["channels"]
        self.pixels = hyperparameters["pixels"]
        self.steps = hyperparameters["steps"]
        self.state_size = hyperparameters["state_size"]
        self.state_depth = hyperparameters["state_depth"]
        self.model = Model(hyperparameters, dtype=self._dtype)
        self.physical_model = physical_model
        # self.grad_scaling_factor = 1/(self.pixels * (self.physical_model.SNR**2 + 1/self.physical_model.sigma**2))
        self.grad_scaling_factor = tf.constant(1/100, dtype)

    @tf.function
    def link_function(self, y):
        return tf.math.log(y + self.noise_floor)

    @tf.function
    def inverse_link_function(self, eta):
        return tf.math.exp(eta)

    # @tf.function
    def grad_scaling(self, grad):
        return grad #tf.clip_by_value(self.grad_scaling_factor * grad, -100, 100)

    def __call__(self, X):
        return self.call(X)

    def call(self, X):
        """
        Method used in training to get model predictions.

        :param X: Vector of complex visibilities amplitude and closure phases
        :return: 5D Tensor of shape (batch_size, [image_size, channels], steps)
        """
        batch_size = X.shape[0]
        y0 = self.initial_guess(batch_size)
        h0 = self.init_hidden_states(batch_size)
        grad = self.physical_model.grad_log_likelihood(y0, X)
        eta_0 = self.link_function(y0)
        grad = self.grad_scaling(grad)
        stacked_input = tf.concat([eta_0, grad], axis=3)
        # compute gradient update
        gt, ht = self.model(stacked_input, h0)
        # update image
        eta_t = eta_0 + gt
        # save output and log likelihood gradient (latter for analysis)
        outputs = tf.expand_dims(eta_t, -1)  # Plus one dimension for step stack
        grads = grad
        for current_step in range(self.steps - 1):
            grad = self.physical_model.grad_log_likelihood(self.inverse_link_function(eta_t), X)
            grad = self.grad_scaling(grad)
            gt, ht = self.model(stacked_input, ht)
            eta_t = eta_t + gt
            outputs = tf.concat([outputs, tf.reshape(eta_t, eta_t.shape + [1])], axis=4)  #TODO use tf.stack
            grads = tf.concat([grads, grad], axis=3)
        return outputs, grads

    def init_hidden_states(self, batch_size):
        return tf.zeros(shape=(batch_size, self.state_size, self.state_size, self.state_depth), dtype=self._dtype)

    def initial_guess(self, batch_size):
        # y0 = self.physical_model.inverse_fourier_transform(X)
        # y0 = softmax_scaler(y0, minimum=0, maximum=1.)
        y0 = tf.ones(shape=[batch_size, self.pixels, self.pixels, 1]) / self.pixels**2
        return y0

    def fit(
            self,
            train_dataset,
            cost_function,
            optimizer,
            max_time,
            metrics=None,
            patience=10,
            track="train_loss",
            checkpoints=5,
            min_delta=0,
            max_epochs=1000,
            test_dataset=None,
            output_dir=None,
            output_save_mod=None,
            checkpoint_dir=None,
            name="rim",
            logdir=None,
            record=True
    ):
        """
        This function trains the weights of the model on the training dataset, which should be a
        tensorflow Dataset class created from zipping X and Y tensors into one dataset. This dataset should have
        been initialised with the enumerate method and batch method. For better performance, use chache and prefetch.
        :param train_dataset: A tensorflow dataset created from zipping X and Y, initialised with batch and enumerate(start=0)
        :param cost_function: target for the training
        :param max_time: Time allowed for training in hours
        :param metrics: A dictionary with metric functions as items that take (Y_pred, Y_true) as input
        :param patience: An integer, the maximum number of epoch without improvement before terminating fit
        :param checkpoints: An integer, correspond to the mod of the epoch at which point the weights are saved
        :param min_delta: This compares loss between epochs; it judges if the loss has been improved if loss - previous_loss <min_delta
        :param max_epochs: Maximum number of epochs allowed
        :param test_dataset: A tensorflow dataset created from zipping X_test and Y_test. Should not be batched (or batch_size = num_samples)
        :param output_dir: directory to store output of the model
        :param output_save_mod: dictionary in the form
            output_save_mod = {
                "index_mod": 25,
                "epoch_mod": 1,
                "step_mod": 1
            },
            each entry correspond to a particular output to be saved. This examples saves the output at
            each epoch and saves each steps, but only saves 1 image per 25 in the dataset. This particular
            examples can produces an enourmous amount of pictures if the user is not careful.
        :param checkpoint_dir: Directory where to save the weights
        :param name: Name of the model
        :return: history, a dictionary with loss and metrics score at each epoch
        """
        if logdir is not None:
            if not os.path.isdir(os.path.join(logdir, "train")):
                os.mkdir(os.path.join(logdir, "train"))
            if not os.path.isdir(os.path.join(logdir, "test")):
                os.mkdir(os.path.join(logdir, "test"))
            train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
            test_writer = tf.summary.create_file_writer(os.path.join(logdir, "test"))
        else:
            test_writer = nullwriter()
            train_writer = nullwriter()
        if record is False:  # TODO remove when summaries are removed from Model
            self.model._timestep_mod = -1
        if metrics is None:
            metrics = {}
        if output_save_mod is None and output_dir is not None:
            output_save_mod = {
                "index_mod": 1,
                "epoch_mod": 1,
                "time_mod": 1,
                "step_mod": 1
            },
        start = time.time()
        if "epoch" in self.hyperparameters.keys():
            _epoch_start = self.hyperparameters["epoch"]
        else:
            _epoch_start = 0
        epoch = 1
        history = {"train_loss": [], "test_loss": []}
        history.update({key + "_train": [] for key in metrics.keys()})
        history.update({key + "_test": [] for key in metrics.keys()})
        min_score = np.inf
        epoch_loss = tf.metrics.Mean()
        _patience = patience
        step = 1  # Counts the number of batches evaluated
        tf.summary.experimental.set_step(step)
        while _patience > 0 and epoch < max_epochs and (time.time() - start) < max_time*3600:
            epoch_loss.reset_states()
            metrics_train = {key: 0 for key in metrics.keys()}
            batch = -1
            with train_writer.as_default():
                for (X, Y) in train_dataset:  # X and Y by ML convention, batch is an index
                    batch += 1
                    with tf.GradientTape() as tape:
                        tape.watch(self.model.trainable_weights)
                        output, grads = self.call(X)
                        cost_value = cost_function(output, self.link_function(Y))
                        cost_value += tf.reduce_sum(self.model.losses)  # Add layer specific regularizer losses (L2 in definitions)
                    epoch_loss.update_state([cost_value])
                    gradient = tape.gradient(cost_value, self.model.trainable_weights)
                    clipped_gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
                    optimizer.apply_gradients(zip(clipped_gradient, self.model.trainable_weights))  # backpropagation

                    # back to image space for analysis
                    output = self.inverse_link_function(output)
                    # ========= Summaries and logs =================
                    tf.summary.scalar("Loss", cost_value, step=step)
                    tf.summary.scalar("Learning rate", optimizer.lr(step).numpy(), step=step)
                    for key, item in metrics_train.items():
                        score = metrics[key](output[..., -1], Y)
                        metrics_train[key] += score.numpy()
                        tf.summary.scalar(key, score, step=step)
                    # dramatically slows down the training if enabled and step_mod too low
                    if record and step % output_save_mod["step_mod"] == 0:
                        tf.summary.histogram(f"Residual_{epoch:04}_{batch:04}", tf.math.log(tf.math.abs(output[..., -1] - Y)), description="Logarithm of the absolute difference between Y_pred and Y")
                        tf.summary.histogram(name=f"Log_Likelihood_Gradient_log_{batch:04}", data=tf.math.log(tf.math.abs(grads[0]))/tf.math.log(10.), description="Log Likelihood gradient for each time steps")
                        tf.summary.histogram(name=f"Log_Likelihood_Gradient{batch:04}", data=grads[0], description="Log Likelihood gradient for each time steps")

                        for i, grad in enumerate(gradient):
                            tf.summary.histogram(name=self.model.trainable_weights[i].name + "_gradient", data=grad, step=step)
                    # independant save of outputs
                    if output_dir is not None:
                        save_output(output, output_dir, epoch + _epoch_start, batch, format="txt", **output_save_mod)
                        save_gradient_and_weights(gradient, self.model.trainable_weights, output_dir, epoch + _epoch_start, batch)
                        save_loglikelihood_grad(grads, output_dir, epoch + _epoch_start, batch, **output_save_mod)
                    train_writer.flush()
                    step += 1
                    tf.summary.experimental.set_step(step)
                    # ================================================
                for key, item in metrics_train.items():
                    history[key + "_train"].append(item/(batch + 1))
                history["train_loss"].append(epoch_loss.result().numpy())

            if test_dataset is not None:
                with test_writer.as_default():
                    for (X, Y) in test_dataset:  # this dataset should not be batched, so this for loop has 1 iteration
                        test_eta_output, _ = self.call(X)
                        test_output = self.inverse_link_function(test_eta_output)
                        test_cost = cost_function(test_eta_output, self.link_function(Y))
                        test_cost += tf.reduce_sum(self.model.losses)
                        history["test_loss"].append(test_cost.numpy())
                        tf.summary.scalar("loss", test_cost, step=step)
                        for key, item in metrics.items():
                            score = item(test_output[..., -1], Y)
                            history[key + "_test"].append(score.numpy())
                            tf.summary.scalar(key, score, step=step)
                        test_writer.flush()
            try:
                print(f"{epoch}: train_loss={history['train_loss'][-1]:.2e} | val_loss={history['test_loss'][-1]:.2e} | "
                      f"learning rate={optimizer.lr(step).numpy():.2e}")
            except IndexError:
                print(f"{epoch}: train_loss={history['train_loss'][-1]:.2e} | learning rate={optimizer.lr(step).numpy():.2e}")
            if history[track][-1] < min_score - min_delta:
                _patience = patience
                min_score = history[track][-1]
            else:
                _patience -= 1
            if checkpoint_dir is not None:
                if epoch % checkpoints == 0 or _patience == 0 or epoch == max_epochs - 1:
                    self.model.save_weights(os.path.join(checkpoint_dir, f"{name}_{epoch + _epoch_start:03}_{cost_value:.5f}.h5"))
            epoch += 1
        self.hyperparameters["epoch"] = epoch + _epoch_start
        return history
