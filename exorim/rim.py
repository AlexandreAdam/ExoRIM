from .definitions import DTYPE
from .utils import nulltape
from .physical_model import PhysicalModel
import tensorflow as tf


class RIM:
    def __init__(self,
                 model,
                 physical_model: PhysicalModel,
                 time_steps=8,
                 adam=True,
                 beta_1=0.9,
                 beta_2=0.99,
                 epsilon=1e-8,
                 log_floor=1e-6 # sets the dynamic range of the model
                 ):
        self.logim = physical_model.logim
        self.pixels = physical_model.pixels
        self.model = model
        self.physical_model = physical_model
        self.steps = time_steps
        self.adam = adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.log_floor = log_floor

        if self.logim:
            self.link_function = tf.keras.layers.Lambda(lambda x: tf.math.log(x + self.log_floor) / tf.math.log(10.))
            self.inverse_link_function = tf.keras.layers.Lambda(lambda x: 10**x)
        else:
            self.link_function = tf.identity
            self.inverse_link_function = tf.identity

        if self.adam:
            self.grad_update = lambda grad, time_step: self.adam_update(grad, time_step)
        else:
            self.grad_update = lambda grad, time_step: grad

    def adam_update(self, grad, time_step):
        self._grad_mean = self. beta_1 * self._grad_mean + (1 - self.beta_1) * grad
        self._grad_var  = self.beta_2 * self._grad_var + (1 - self.beta_2) * tf.square(grad)
        m_hat = self._grad_mean / (1 - self.beta_1**(time_step + 1)) # Unbias moments
        v_hat = self._grad_var / (1 - self.beta_2**(time_step + 1))
        return m_hat / (tf.sqrt(v_hat) + self.epsilon)

    def initial_states(self, batch_size):
        source = tf.zeros(shape=[batch_size, self.pixels, self.pixels, 1])
        states = self.model.init_hidden_states(input_pixels=self.pixels, batch_size=batch_size)
        self._grad_mean = tf.zeros_like(source)
        self._grad_var = tf.zeros_like(source)
        return source, states

    def __call__(self, X, outer_tape=nulltape):
        return self.call(X, outer_tape)

    def call(self, X, outer_tape=nulltape):
        """
        Method used in training to get model predictions.

        :param X: Vector of complex visibilities amplitude and closure phases.
        :return: 5D Tensor of shape (steps, batch_size, pixels, pixels, channels)
        """
        batch_size = X.shape[0]
        xt, ht = self.initial_states(batch_size)

        image_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with outer_tape.stop_recording():
                grad = self.physical_model.grad_log_likelihood(image=xt, X=X)
                grad = self.grad_update(grad, time_step=current_step)
            xt, ht = self.model(xt, ht, grad)
            image_series = image_series.write(index=current_step, value=xt)
        return image_series.stack()


    # def fit(
    #         self,
    #         train_dataset,
    #         cost_function,
    #         optimizer,
    #         max_time=np.inf,
    #         metrics=None,
    #         patience=np.inf,
    #         track="train_loss",
    #         min_delta=0,
    #         max_epochs=10,
    #         test_dataset=None,
    #         output_dir=None,
    #         output_save_mod=None,
    #         checkpoint_manager=None,
    #         checkpoints=10, # some integer in case checkpoint_manager is not None
    #         name="rim",
    #         logdir=None,
    #         record=False
    # ):
    #     """
    #     This function trains the weights of the model on the training dataset, which should be a
    #     tensorflow Dataset class created from zipping X and Y tensors into one dataset. This dataset should have
    #     been initialised with the enumerate method and batch method. For better performance, use chache and prefetch.
    #     :param train_dataset: A tensorflow dataset created from zipping X and Y, initialised with batch and enumerate(start=0)
    #     :param cost_function: target for the training
    #     :param max_time: Time allowed for training in hours
    #     :param metrics: A dictionary with metric functions as items that take (Y_pred, Y_true) as input
    #     :param patience: An integer, the maximum number of epoch without improvement before terminating fit
    #     :param checkpoints: An integer, correspond to the mod of the epoch at which point the weights are saved
    #     :param min_delta: This compares loss between epochs; it judges if the loss has been improved if loss - previous_loss <min_delta
    #     :param max_epochs: Maximum number of epochs allowed
    #     :param test_dataset: A tensorflow dataset created from zipping X_test and Y_test. Should not be batched (or batch_size = num_samples)
    #     :param output_dir: directory to store output of the model
    #     :param output_save_mod: dictionary in the form
    #         output_save_mod = {
    #             "index_mod": 25,
    #             "epoch_mod": 1,
    #             "step_mod": 1
    #         },
    #         each entry correspond to a particular output to be saved. This examples saves the output at
    #         each epoch and saves each steps, but only saves 1 image per 25 in the dataset. This particular
    #         examples can produces an enourmous amount of pictures if the user is not careful.
    #     :param checkpoint_dir: Directory where to save the weights
    #     :param name: Name of the model
    #     :return: history, a dictionary with loss and metrics score at each epoch
    #     """
    #     if logdir is not None:
    #         if not os.path.isdir(os.path.join(logdir, "train")):
    #             os.mkdir(os.path.join(logdir, "train"))
    #         if not os.path.isdir(os.path.join(logdir, "test")):
    #             os.mkdir(os.path.join(logdir, "test"))
    #         train_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
    #         test_writer = tf.summary.create_file_writer(os.path.join(logdir, "test"))
    #     else:
    #         test_writer = nullwriter()
    #         train_writer = nullwriter()
    #     if record is False:  # TODO remove when summaries are removed from Model
    #         self.model._timestep_mod = -1
    #     if metrics is None:
    #         metrics = {}
    #     if output_save_mod is None:
    #         output_save_mod = {
    #             "index_mod": -1,
    #             "epoch_mod": -1,
    #             "time_mod": -1,
    #             "step_mod": -1
    #         }
    #     if checkpoint_manager is not None:
    #         checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint)
    #     start = time.time()
    #     epoch = 1
    #     history = {"train_loss": [], "test_loss": []}
    #     history.update({key + "_train": [] for key in metrics.keys()})
    #     history.update({key + "_test": [] for key in metrics.keys()})
    #     min_score = np.inf
    #     epoch_loss = tf.metrics.Mean()
    #     _patience = patience
    #     step = 1  # Counts the number of batches evaluated
    #     tf.summary.experimental.set_step(step)
    #     while _patience > 0 and epoch < max_epochs and (time.time() - start) < max_time*3600:
    #         epoch_loss.reset_states()
    #         metrics_train = {key: 0 for key in metrics.keys()}
    #         batch = 0
    #         with train_writer.as_default():
    #             for (X, Y) in train_dataset:  # X and Y by ML convention, batch is an index
    #                 batch += 1
    #                 with tf.GradientTape() as tape:
    #                     tape.watch(self.model.trainable_weights)
    #                     output, grads = self.call(X)
    #                     cost_value = cost_function(output, self.link_function(Y))
    #                     cost_value += tf.reduce_sum(self.model.losses)  # Add layer specific regularizer losses (L2 in definitions)
    #                 epoch_loss.update_state([cost_value])
    #                 gradient = tape.gradient(cost_value, self.model.trainable_weights)
    #                 clipped_gradient = [tf.clip_by_value(grad, -10, 10) for grad in gradient]
    #                 optimizer.apply_gradients(zip(clipped_gradient, self.model.trainable_weights))  # backpropagation
    #
    #                 # back to image space for analysis
    #                 output = self.inverse_link_function(output)
    #                 # ========= Summaries and logs =================
    #                 tf.summary.scalar("Loss", cost_value, step=step)
    #                 tf.summary.scalar("Learning rate", optimizer.lr(step).numpy(), step=step)
    #                 for key, item in metrics_train.items():
    #                     score = metrics[key](output[..., -1], Y)
    #                     metrics_train[key] += score.numpy()
    #                     tf.summary.scalar(key, score, step=step)
    #                 # dramatically slows down the training if enabled and step_mod too low
    #                 if record and step % output_save_mod["step_mod"] == 0:
    #                     tf.summary.histogram(f"Residual_first", tf.math.log(tf.math.abs(output[..., 0] - Y)),
    #                         description="Logarithm of the absolute difference between Y_pred and Y at the first time step")
    #                     tf.summary.histogram(f"Residual_last", tf.math.log(tf.math.abs(output[..., -1] - Y)),
    #                         description="Logarithm of the absolute difference between Y_pred and Y at the last time step")
    #                     if not self.grad_log_scale:
    #                         tf.summary.histogram(name=f"Log_Likelihood_Gradient_log_scale", data=tf.math.asinh(grads[0])/tf.math.log(10.),
    #                             description="Arcsinh of the Likelihood gradient for each time steps (divided by log(10))")
    #                     tf.summary.histogram(name=f"Log_Likelihood_Gradient", data=grads[0], description="Log Likelihood gradient")
    #                     tf.summary.image(name="Training batch prediction", data=output[..., -1])
    #                     tf.summary.image(name="Training batch labels",     data=self.link_function(Y))
    #                     for i, grad in enumerate(gradient):
    #                         tf.summary.histogram(name=self.model.trainable_weights[i].name + "_gradient", data=grad, step=step)
    #
    #                     if output_dir is not None:
    #                         save_output(output, output_dir, epoch, batch, format="txt", **output_save_mod)
    #                         save_gradient_and_weights(gradient, self.model.trainable_weights, output_dir, epoch, batch)
    #                         save_loglikelihood_grad(grads, output_dir, epoch, batch, **output_save_mod)
    #                 train_writer.flush()
    #                 step += 1
    #                 tf.summary.experimental.set_step(step)
    #                 # ================================================
    #             for key, item in metrics_train.items():
    #                 history[key + "_train"].append(item/(batch + 1))
    #             history["train_loss"].append(epoch_loss.result().numpy())
    #
    #         if test_dataset is not None:
    #             with test_writer.as_default():
    #                 for (X, Y) in test_dataset:  # this dataset should not be batched, so this for loop has 1 iteration
    #                     test_eta_output, _ = self.call(X)
    #                     test_output = self.inverse_link_function(test_eta_output)
    #                     test_cost = cost_function(test_eta_output, self.link_function(Y))
    #                     test_cost += tf.reduce_sum(self.model.losses)
    #                     history["test_loss"].append(test_cost.numpy())
    #                     tf.summary.scalar("loss", test_cost, step=step)
    #                     for key, item in metrics.items():
    #                         score = item(test_output[..., -1], Y)
    #                         history[key + "_test"].append(score.numpy())
    #                         tf.summary.scalar(key, score, step=step)
    #                     test_writer.flush()
    #         try:
    #             print(f"{epoch}: train_loss={history['train_loss'][-1]:.2e} | val_loss={history['test_loss'][-1]:.2e} | "
    #                   f"learning rate={optimizer.lr(step).numpy():.2e}")
    #         except IndexError:
    #             print(f"{epoch}: train_loss={history['train_loss'][-1]:.2e} | learning rate={optimizer.lr(step).numpy():.2e}")
    #         if history[track][-1] < min_score - min_delta:
    #             _patience = patience
    #             min_score = history[track][-1]
    #         else:
    #             _patience -= 1
    #         if checkpoint_manager is not None:
    #             checkpoint_manager.checkpoint.step.assign_add(1) # a bit of a hack
    #             # save a model if checkpoint is reached and model has improved or last step has been reached
    #             if epoch % checkpoints == 0 or _patience == 0 or epoch == max_epochs - 1:
    #                 checkpoint_manager.save() # os.path.join(checkpoint_dir, f"{name}_{epoch:03}_{cost_value:.5f}.h5")
    #                 print("Saved checkpoint for step {}: {}".format(int(checkpoint_manager.checkpoint.step), checkpoint_manager.latest_checkpoint))
    #         epoch += 1
    #     return history
