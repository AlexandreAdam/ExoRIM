import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from datetime import datetime

dtype = tf.float32
AUTOTUNE = tf.data.experimental.AUTOTUNE
checkpoint_dir = "../models/ideal_fft_forward/"
results_dir = "../results/ideal_fft_forward/"
logs = "../logs/ideal_fft_forward/"


class Preprocessing:
    def __init__(self, pixels):
        self.pixels = pixels

    def transform(self, X):
        return X


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

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_pred = tf.reshape(y_pred, [-1, self.pixels, self.pixels, 1])
        y_true = tf.reshape(y_true, [-1, self.pixels, self.pixels, 1])
        self.mean.update_state(tf.image.ssim(y_true, y_pred, max_val=1./self.pixels**2))

    def reset_states(self):
        self.mean.reset_states()

    def result(self):
        return self.mean.result()


class Generator:

    def __init__(self, number_images, pixels, seed=0, fixed=False):
        self.pixels = pixels
        self.number_images = number_images
        self.epoch = seed
        self.fixed = fixed
        self.transform = Preprocessing(pixels).transform

    def generator(self):
        self.epoch += 1
        if not self.fixed:
            np.random.seed(self.epoch)
        for i in range(self.number_images):
            a, b = np.random.choice([1, 2], size=2)
            sigma = np.random.uniform(1, self.pixels/6, size=1)
            image = blob(self.pixels, sigma=sigma, a=a, b=b)
            image /= image.sum()
            X = ideal_forward_model(image)
            Y = tf.constant(image.flatten(), dtype)
            yield self.transform(X), Y


def input_shape(pixels):
    # return 2 * pixels * (pixels//2 + 1) # without zero pad
    return 4 * pixels * (pixels + 1)  # because of zero pad


def build_model(pixels):
    # input_shape = 2*pixels*(pixels//2 + 1)
    output_shape = pixels**2
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape(pixels)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(output_shape, activation="softmax",
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
    return model


def dataset(number_images, pixels, batch_size, seed=42, fixed=False):
    gen = Generator(number_images, pixels, seed, fixed)
    shapes = (tf.TensorShape([input_shape(pixels)]), tf.TensorShape([pixels**2]))
    D = tf.data.Dataset.from_generator(gen.generator, output_types=(dtype, dtype), output_shapes=shapes)
    D = D.cache()              # accelerate the second and subsequent iterations over the dataset
    D = D.batch(batch_size, drop_remainder=True)
    D = D.prefetch(AUTOTUNE)  # Batch is prefetched
    return D


def ideal_forward_model(image):
    pixels = image.shape[0]
    paddings = tf.constant([[pixels//2, pixels//2], [pixels//2, pixels//2]])
    image = tf.pad(image, paddings, "CONSTANT")
    image_hat = tf.signal.fftshift(tf.signal.rfft2d(image))  # rfft to get rid of negative frequencies for real input
    image_hat = tf.reshape(image_hat, [-1])  # flatten
    out = tf.concat([tf.math.real(image_hat), tf.math.imag(image_hat)], axis=0)
    return tf.cast(out, dtype)


def blob(pixels, sigma, x0=0, y0=0, a=1, b=1):
    x = np.arange(pixels) - pixels/2
    xx, yy = np.meshgrid(x, x)
    image = np.zeros_like(xx)
    rho = np.sqrt((xx - x0)**2/a + (yy - y0)**2/b)
    image += np.exp(-(rho/sigma)**2)
    return image


def main():
    pixels = 32
    date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1e-1,
        decay_steps=1000,
        end_learning_rate=1e-5
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model = build_model(pixels)
    # model.load_weights(checkpoint_dir)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss, metrics=[SSIM(pixels)])
    model.summary()
    train_dataset = dataset(1000, pixels, batch_size=20)
    valid_dataset = dataset(100, pixels, batch_size=100, seed=31415926, fixed=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_SSIM", patience=3, mode="max", min_delta=0.001)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_dir,
        monitor="val_SSIM",
        save_weights_only=True,
        mode='max',
        save_best_only=True
    )
    os.mkdir(os.path.join(logs, date))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logs, date), update_freq="batch")
    # ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
    # manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    model.fit(train_dataset,
              epochs=100,
              callbacks=[early_stopping, checkpoint, LearningRateTracker(opt), tensorboard],
              validation_data=valid_dataset)
    valid_dataset = dataset(100, pixels, batch_size=1, seed=31415926, fixed=True)
    for batch, (X, Y) in enumerate(valid_dataset):
        if batch % 25 != 0:
            continue
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        Y_pred = model(X).numpy()[0].reshape((pixels, pixels))
        Y_true = Y.numpy()[0].reshape((pixels, pixels))
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
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(logs):
        os.mkdir(logs)
    main()



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