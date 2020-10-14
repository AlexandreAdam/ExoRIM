import numpy as np
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

dtype = tf.float64
AUTOTUNE = tf.data.experimental.AUTOTUNE
checkpoint_dir = "../models/ideal_fft_forward/"
results_dir = "../results/ideal_fft_forward/"
logs = "../logs/ideal_fft_forward/"
flower_data = "../data/flower/"
pixels = 64
batch_size = 16
epochs = 10
natural = True


class Preprocessing:
    def __init__(self, pixels, p=0.6):
        self.pixels = pixels
        # self.mask = tf.constant(bernoulli.rvs(p, 0, pixels**2), dtype)
        self.mask = np.ones([pixels**2])

    def transform(self, X):
        amp = tf.math.abs(X)
        phi = tf.math.angle(X)
        X = tf.concat([amp * self.mask,
                       tf.math.cos(phi) * self.mask,
                       tf.math.sin(phi) * self.mask], axis=0)
        # X = tf.concat([tf.math.real(X), tf.math.real(X)], axis=0)
        return X

    def input_shape(self):
        return 3 * self.pixels ** 2


class LearningRateTracker(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
        super(LearningRateTracker, self).__init__()
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, *args, **kwargs):
        lr = self.optimizer.lr(self.optimizer.iterations)
        print(' | learning rate: {:.2e}\n'.format(lr))


class SSIM(tf.metrics.Metric):
    def __init__(self, pixels, name="SSIM"):
        super(SSIM, self).__init__(name=name, dtype=dtype)
        self.mean = tf.keras.metrics.Mean()
        self.pixels = pixels
        if natural:
            self.max_val = 1.0
        else:
            self.max_val = 1/100  # approximation

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_pred = tf.reshape(y_pred, [-1, self.pixels, self.pixels, 1])
        y_true = tf.reshape(y_true, [-1, self.pixels, self.pixels, 1])
        self.mean.update_state(tf.image.ssim(y_true, y_pred, max_val=self.max_val))

    def reset_states(self):
        self.mean.reset_states()

    def result(self):
        return self.mean.result()


class Generator:

    def __init__(self, number_images, seed=0, fixed=False):
        self.number_images = number_images
        self.epoch = seed
        self.fixed = fixed
        self.preprocess = Preprocessing(pixels)

    def generator(self):
        self.epoch += 1
        if not self.fixed:
            np.random.seed(self.epoch)
        for i in range(self.number_images):
            n = int(np.random.choice(list(range(1, 10)), 1))
            intensity = np.random.uniform(0.6, 1, n)
            abs = np.random.uniform(1, 3, size=(n, 2))
            xys = np.random.uniform(-10, 10, size=(n, 2))
            sigmas = np.random.uniform(3, 7, size=n)
            image = intensity[0] * blob(sigma=sigmas[0], a=abs[0, 0], b=abs[0, 0])
            for i in range(1, n):
                image += intensity[i] * blob(sigmas[i], *xys[i], *abs[i])
            # image /= image.sum()
            X = ideal_forward_model(image)
            Y = tf.constant(image.flatten(), dtype)
            yield self.preprocess.transform(X), Y


def flower_dataset():
    files = list(os.walk(flower_data))[0][2]
    images = []
    data = []
    for file in files:
        im = np.asarray(Image.open(os.path.join(flower_data, file)))
        im = tf.image.rgb_to_grayscale(im)
        Y = tf.image.resize(im, [pixels, pixels]) / 255
        X = Preprocessing(pixels).transform(ideal_forward_model(Y))
        Y = tf.reshape(Y, [pixels**2])
        images.append(Y)
        data.append(X)
    x = tf.data.Dataset.from_tensor_slices(data)
    y = tf.data.Dataset.from_tensor_slices(images)
    dataset = tf.data.Dataset.zip((x, y))
    dataset = dataset.cache()              # accelerate the second and subsequent iterations over the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched
    return dataset


def build_model():
    # input_shape = 2*pixels*(pixels//2 + 1)
    output_shape = pixels**2
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=Preprocessing(pixels).input_shape()))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dense(output_shape, activation="relu", use_bias=True, kernel_regularizer=tf.keras.regularizers.l2()))
    # model.add(tf.keras.layers.Dense(output_shape, activation="relu", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2()))
    model.add(tf.keras.layers.Dense(output_shape, activation="linear", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2()))
    return model


def cnn_model():
    output_shape = pixels**2
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=Preprocessing(pixels).input_shape()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(output_shape, activation="relu", use_bias=True, kernel_regularizer=tf.keras.regularizers.l2()))
    model.add(tf.keras.layers.Reshape([pixels, pixels, 1]))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", kernel_regularizer=tf.keras.regularizers.l2()))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", kernel_regularizer=tf.keras.regularizers.l2()))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Reshape([output_shape]))
    return model


def dataset(number_images, pixels, batch_size, seed=42, fixed=False):
    gen = Generator(number_images, seed, fixed)
    shapes = (tf.TensorShape([Preprocessing(pixels).input_shape()]), tf.TensorShape([pixels**2]))
    D = tf.data.Dataset.from_generator(gen.generator, output_types=(dtype, dtype), output_shapes=shapes)
    D = D.cache()              # accelerate the second and subsequent iterations over the dataset
    D = D.batch(batch_size, drop_remainder=True)
    D = D.prefetch(AUTOTUNE)  # Batch is prefetched
    return D


def ideal_forward_model(image):
    pixels = image.shape[0]
    h = pixels // 4
    # paddings = tf.constant([[h, h], [h, h]])
    # image = tf.pad(image, paddings, "CONSTANT")
    image_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(image, tf.complex128)))
    image_hat = tf.reshape(image_hat, [-1])  # flatten
    return image_hat


def blob(sigma, x0=0, y0=0, a=1, b=1):
    x = np.arange(pixels) - pixels/2
    xx, yy = np.meshgrid(x, x)
    image = np.zeros_like(xx)
    rho = np.sqrt((xx - x0)**2/a + (yy - y0)**2/b)
    image += np.exp(-(rho/sigma)**2)
    return image


def main():
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    # lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    #     initial_learning_rate=1e-1,
    #     decay_steps=500,
    #     end_learning_rate=1e-6
    # )
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=1e-2,
        decay_steps=10,
        decay_rate=20,
        staircase=False
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # model = build_model()
    model = cnn_model()

    model.load_weights(checkpoint_dir)

    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss, metrics=[SSIM(pixels)])
    model.summary()
    if natural:
        train_dataset = flower_dataset()
        valid_dataset = None
    else:
        train_dataset = dataset(1000, pixels, batch_size=20)
        valid_dataset = dataset(100, pixels, batch_size=100, seed=31415926, fixed=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2, mode="min", min_delta=1e-4)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_dir,
        monitor="loss",
        save_weights_only=True,
        mode='min',
        save_best_only=True
    )
    os.mkdir(os.path.join(logs, date))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logs, date), update_freq="batch")
    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
    # manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    model.fit(train_dataset,
              epochs=epochs,
              callbacks=[early_stopping, checkpoint, LearningRateTracker(opt), tensorboard],
              validation_data=valid_dataset)

    if natural:
        valid_dataset = train_dataset
    else:
        valid_dataset = dataset(100, pixels, batch_size=1, seed=0, fixed=True)
    for batch, (X, Y) in enumerate(valid_dataset):
        if batch % 25 != 0:
            continue
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle("Retrained with p = 0.9")
        Y_pred = model(X).numpy()[0].reshape((pixels, pixels))
        Y_true = Y.numpy()[0].reshape((pixels, pixels))
        ax1.imshow(Y_pred, cmap="gray")
        ax1.set_title("Y_pred")
        ax1.axis("off")
        ax2.imshow(Y_true, cmap="gray")
        ax2.set_title("Y_true")
        ax2.axis("off")
        im = ax3.imshow(np.log(np.abs(Y_pred - Y_true)), cmap="coolwarm")
        ax3.set_title("Log Residual")
        ax3.axis("off")
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(results_dir, f"image_{batch:02}.png"))
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 9))
    # im = ax1.imshow(model.get_weights()[-2][0].reshape((pixels, pixels)), cmap="coolwarm")
    # ax1.axis("off")
    # ax1.set_title("Weight for first input node")
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    # im = ax2.imshow(model.get_weights()[-2][1023].reshape((pixels, pixels)),  cmap="coolwarm")
    # ax2.axis("off")
    # ax2.set_title("Weight for mid input node")
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    # im = ax3.imshow(model.get_weights()[-2][-1].reshape((pixels, pixels)),  cmap="coolwarm")
    # ax3.axis("off")
    # ax3.set_title("Weight for last input node")
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    # plt.savefig(os.path.join(results_dir, "model_weights.png"))


def debug():
    gen = Generator(10, 32)
    for X, Y in gen.generator():
        pass


def analyze_results():
    model = cnn_model()
    model.load_weights(checkpoint_dir)
    p1 = 0.3
    p2 = 0.5
    p3 = 0.9
    p4 = 1
    mask1 = np.concatenate([bernoulli.rvs(p1, 0, pixels**2)]*3)
    mask2 = np.concatenate([bernoulli.rvs(p2, 0, pixels**2)]*3)
    mask3 = np.concatenate([bernoulli.rvs(p3, 0, pixels**2)]*3)
    mask4 = np.concatenate([bernoulli.rvs(p4, 0, pixels**2)]*3)

    files = list(os.walk(flower_data))[0][2]
    images = []
    data = []
    for file in files:
        im = np.asarray(Image.open(os.path.join(flower_data, file)))
        im = tf.image.rgb_to_grayscale(im)
        Y = tf.image.resize(im, [pixels, pixels]) / 255
        X = Preprocessing(pixels).transform(ideal_forward_model(Y))
        Y = tf.reshape(Y, [pixels**2])
        images.append(Y)
        data.append(X)
    x = tf.data.Dataset.from_tensor_slices(data)
    y = tf.data.Dataset.from_tensor_slices(images)
    z = tf.data.Dataset.zip((x, y))
    z = z.batch(1)
    for batch, (X, Y) in enumerate(z):
        if batch % 25 != 0:
            continue
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        Y_pred1 = model(mask1 * X).numpy()[0].reshape((pixels, pixels))
        Y_pred2 = model(mask2 * X).numpy()[0].reshape((pixels, pixels))
        Y_pred3 = model(mask3 * X).numpy()[0].reshape((pixels, pixels))
        # Y_pred4 = model(mask1 * X).numpy()[0].reshape((pixels, pixels))

        Y_true = Y.numpy()[0].reshape((pixels, pixels))
        ax1.imshow(Y_pred1, cmap="gray")
        ax1.set_title(f"p = {p1}")
        ax1.axis("off")
        ax2.imshow(Y_pred2, cmap="gray")
        ax2.set_title(f"p = {p2}")
        ax2.axis("off")
        ax3.imshow(Y_pred3, cmap="gray")
        ax3.set_title(f"p = {p3}")
        ax3.axis("off")
        ax4.imshow(Y_true, cmap="gray")
        ax4.set_title(f"Ground Truth")
        ax4.axis("off")

        # divider = make_axes_locatable(ax3)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        plt.savefig(os.path.join(results_dir, f"analysis_{batch:02}.png"))


if __name__ == '__main__':
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(logs):
        os.mkdir(logs)
    # main()
    # debug()
    # flower_dataset()
    analyze_results()


# A custom loop if needed

# def train_and_checkpoint(net, manager, checkpoint, iterator, opt, max_step, valid_dataset=None):
#   checkpoint.restore(manager.latest_checkpoint)
#   if manager.latest_checkpoint:
#     print("Restored from {}".format(manager.latest_checkpoint))
#   else:
#     print("Initializing from scratch.")
#
#   for _ in range(max_step):
#     X, Y = next(iterator)
#     loss = train_step(net, X, Y, opt)
#     checkpoint.step.assign_add(1)
#     if int(checkpoint.step) % 50 == 0:
#       save_path = manager.save()
#       print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
#       print("loss {:1.2f}".format(loss.numpy()))
#
#
# def train_step(net, X, Y, optimizer):
#     with tf.GradientTape() as tape:
#         output = net(X)
#         loss = tf.reduce_mean(tf.square(output - Y))
#     variables = net.trainable_variables
#     gradients = tape.gradient(loss, variables)
#     optimizer.apply_gradients(zip(gradients, variables))
#     return loss