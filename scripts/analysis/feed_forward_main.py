import tensorflow as tf
import numpy as np
from exorim.definitions import dtype
from exorim.models import FeedForwardModel
from exorim.physical_model import PhysicalModel
# from exorim.operators import Baselines, closure_fourier_matrices, redundant_phase_closure_operator
# from exorim.utilities import create_dataset_from_generator, replay_dataset_from_generator
from exorim.loss import Loss, MAE
import exorim.log_likelihood as chisq
import json, os
from datetime import datetime
import matplotlib.pyplot as plt
import pdb

AUTOTUNE = tf.data.experimental.AUTOTUNE


class SimpleGenerator:

    def __init__(self, number_images, pixels, phys, seed=42):
        self.pixels = pixels
        self.number_images = number_images
        self.phys = phys
        np.random.seed(seed)

    def generator(self):
        smallest_scale = self.phys.smallest_scale
        largest_scale = self.pixels // 4
        x = np.arange(self.pixels) - self.pixels // 2
        xx, yy = np.meshgrid(x, x)
        zeros = np.zeros_like(xx, dtype=np.float32)
        coord_pool = np.arange(-self.pixels // 2 + largest_scale // 2, self.pixels // 2 - largest_scale // 2)
        for index in range(self.number_images):
            image = zeros
            # x0, y0 = np.random.choice(coord_pool, size=2)
            x0 = 0
            y0 = 0
            a, b = np.random.uniform(1, 3, size=2)
            width = np.random.uniform(smallest_scale, largest_scale, 1)
            rho = np.sqrt((xx - x0)**2/a + (yy - y0)**2/b)
            image += np.exp(-(rho / width)**2)
            image /= image.sum()
            Y = tf.constant(image.reshape([1, self.pixels, self.pixels, 1]), dtype)
            X = tf.squeeze(self.phys.forward(Y))
            yield X, tf.squeeze(Y, axis=0)


def simple_dataset(number_images, pixels, phys, batch_size):
    gen = SimpleGenerator(number_images, pixels, phys)
    # shapes = (tf.TensorShape([phys.q + phys.p]), tf.TensorShape([pixels, pixels, 1]))
    shapes = (tf.TensorShape([2*phys.p]), tf.TensorShape([pixels, pixels, 1]))
    dataset = tf.data.Dataset.from_generator(gen.generator, output_types=(dtype, dtype), output_shapes=shapes)
    dataset = dataset.cache()              # accelerate the second and subsequent iterations over the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs
    return dataset


def main():
    with open("../hyperparameters_feedforward.json", "r") as file:
        hparams = json.load(file)
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    checkpoint_dir = "../models/feed_forward/" + date
    results_dir = "../results/feed_forward/" + date
    os.mkdir(results_dir)

    data_dir = "../data/feed_forward/" + date
    logs = "../logs/FeedForward/" + date

    N = 21
    pq = N * (N - 1) // 2 + (N - 1) * (N - 2) // 2
    batch = 10
    mask = np.random.normal(0, 6, (N, 2))
    phys = PhysicalModel(pixels=hparams["pixels"], mask_coordinates=mask)
                         # chisq_term="visibility", x_transform="append_real_imag_visibility")#append_amp_closure_phase")
    # dataset = create_dataset_from_generator(phys, item_per_epoch=1000, batch_size=batch)
    # valid_data = create_dataset_from_generator(phys, item_per_epoch=100, batch_size=10, fixed=True, seed=31415)
    dataset = simple_dataset(number_images=1000, pixels=hparams["pixels"], batch_size=50, phys=phys)
    valid_data = simple_dataset(number_images=100, pixels=hparams["pixels"], batch_size=100, phys=phys)
    model = FeedForwardModel(hparams, name="FeedForward")
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1e-1,
        decay_steps=100,
        end_learning_rate=1e-3,
        power=2
    )
    mae = MAE()
    metrics = {
        "ssim": lambda Y_pred, Y_true: tf.reduce_mean(tf.image.ssim(Y_pred, Y_true, max_val=1.0)),
        "mae": lambda Y_pred, Y_true: mae.test(Y_pred, Y_true)
    }
    metrics.update({
        "Chi_squared_visibilities": lambda Y_pred, Y_true: tf.reduce_mean(chisq.chi_squared_complex_visibility(
            Y_pred, phys.forward(Y_true)[:, :phys.p], phys
        )),
        "Chi_squared_closure_phases": lambda Y_pred, Y_true: tf.reduce_mean(chisq.chi_squared_closure_phasor(
            Y_pred, tf.math.angle(phys.bispectrum(Y_true)), phys
        )),
        "Chi_squared_amplitude": lambda Y_pred, Y_true: tf.reduce_mean(chisq.chi_squared_amplitude(
            Y_pred, tf.math.abs(phys.forward(Y_true)), phys
        ))
    })
    loss = tf.keras.losses.MSE
    # loss = tf.keras.losses.KLDivergence()
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss=loss)
    model(tf.random.normal(shape=[batch, 2*phys.p]))
    model.summary()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, update_freq='batch')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=1e-10, patience=3)

    os.mkdir(checkpoint_dir)
    os.mkdir(logs)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model_{epoch:02d}-{val_loss:.2f}.tf")
    )
    model.fit(dataset, callbacks=[tensorboard, early_stopping, checkpoint], validation_data=valid_data, epochs=20)
    # replay_dataset_from_generator(valid_data, epochs=1, dirname=data_dir, format="txt", fixed=True, index_mod=25)
    valid_data = simple_dataset(number_images=100, pixels=hparams["pixels"], batch_size=1, phys=phys)
    for batch, (X, Y) in enumerate(valid_data):
        if batch % 25 != 0:
            continue
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        Y_pred = model(X).numpy()[0, ..., 0]
        Y_true = Y.numpy()[0, ..., 0]
        ax1.imshow(Y_pred, cmap="gray")
        ax1.set_title("Y_pred")
        ax1.axis("off")
        ax2.imshow(Y_true, cmap="gray")
        ax2.set_title("Y_true")
        ax2.axis("off")
        ax3.imshow(np.abs(Y_pred - Y_true), cmap="gray")
        ax3.set_title("Residual")
        ax3.axis("off")
        plt.savefig(os.path.join(results_dir, f"image_{batch:02}.png"))


if __name__ == '__main__':
    if not os.path.isdir("../models/feed_forward"):
        os.mkdir("../models/feed_forward")
    if not os.path.isdir("../results/feed_forward"):
        os.mkdir("../results/feed_forward")
    if not os.path.isdir("../logs/FeedForward"):
        os.mkdir("../logs/FeedForward")
    if not os.path.isdir("../data/feed_forward/"):
        os.mkdir("../data/feed_forward/")
    main()
    # mask = np.random.normal(0, 6, (10, 2))
    # phys = PhysicalModel(pixels=32, mask_coordinates=mask,
    #                      chisq_term="visibility", x_transform="append_amp_closure_phase")
    # gen = SimpleGenerator(10, 32, phys)
    # for X, Y in gen.generator():
    #     pass