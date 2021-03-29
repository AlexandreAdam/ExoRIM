import tensorflow as tf
from exorim.models.modelv1 import Model
from exorim.rim import RIM
from exorim.interferometry.models.direct_fourier_transform import PhysicalModel
from exorim.interferometry.simulated_data import CenteredBinaries
from exorim.loss import MSE
import json
import numpy as np

def test_hparam():
    image = tf.random.uniform(shape=(5, 32, 32, 1))
    phys = PhysicalModel(32)
    X = phys.forward(image)
    rim = RIM(phys)(X)

    rim = RIM(
        phys,
        time_steps=6,
        state_depth=32,
        grad_log_scale=True,
        batch_norm=True,
        kernel_size_downsampling=3,
        filters_downsampling=16,
        downsampling_layers=2,
        conv_layers=1,
        kernel_size_gru=3,
        hidden_layers=1,
        filters_upsampling=16,
        activation="relu",
        kernel_regularizer_amp=1e-3,
        bias_regularizer_amp=1e-1
    )(X)

    rim = RIM(
        phys,
        time_steps=6,
        state_depth=64,
        grad_log_scale=True,
        kernel_size_downsampling=3,
        filters_downsampling=16,
        downsampling_layers=1,
        conv_layers=2,
        kernel_size_gru=3,
        hidden_layers=1,
        filters_upsampling=16,
        activation="relu",
        kernel_regularizer_amp=1e-3,
        bias_regularizer_amp=1e-1
    )(X)

    rim = RIM(
        phys,
        time_steps=6,
        state_depth=16,
        grad_log_scale=True,
        kernel_size_downsampling=3,
        filters_downsampling=16,
        downsampling_layers=2,
        conv_layers=1,
        kernel_size_gru=3,
        hidden_layers=2,
        filters_upsampling=32,
        activation="relu",
        kernel_regularizer_amp=1e-3,
        bias_regularizer_amp=1e-1
    )(X)

# TODO rework these test since hparam file is no longer supported
# def test_call():
#     with open("../hyperparameters_small.json", "r") as file:
#         hparams = json.load(file)
#     pix = hparams["pixels"]
#     state_size = hparams["state_size"]
#     state_depth = hparams["state_depth"]
#     X = tf.random.normal(shape=(10, pix, pix, 1))
#     h = tf.zeros(shape=(state_size, state_size, state_depth))
#     model = Model(hparams)
#     pred = model.call(X, h)



# def test_save_and_load():
#     with open("../hyperparameters_small.json", "r") as file:
#         hparams = json.load(file)
#     pix = hparams["pixels"]
#     state_size = hparams["state_size"]
#     state_depth = hparams["state_depth"]
#     X = tf.random.normal(shape=(10, pix, pix, 1))
#     h = tf.zeros(shape=(10, state_size, state_size, state_depth))
#     model = Model(hparams)
#     model(X, h)
#     model.save_weights("model_test.h5")
#
#     model = Model(hparams)
#     model(X, h)
#     model.load_weights("model_test.h5")
#
#
# def test_rim_fit():
#     with open("../hyperparameters_small.json", "r") as file:
#         hparams = json.load(file)
#     mask = np.random.normal(0, 1, (12, 2))
#     phys = PhysicalModel(pixels=32, mask_coordinates=mask, wavelength=0.5e-6, SNR=10)
#     rim = RIM(physical_model=phys, hyperparameters=hparams, noise_floor=0.1)
#
#     train_meta = CenteredBinaries(total_items=10, pixels=32, width=1)
#     images = tf.constant(train_meta.generate_epoch_images(), tf.float32)
#     noisy_data = rim.physical_model.forward(images)  # TODO make this noisy forward
#     X = tf.data.Dataset.from_tensor_slices([noisy_data[i] for i in range(10)])  # split along batch dimension
#     Y = tf.data.Dataset.from_tensor_slices([images[i] for i in range(10)])
#     dataset = tf.data.Dataset.zip((X, Y))
#     dataset = dataset.batch(2)
#
#     learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 10, 0.9)
#     cost_func = MSE()
#     optimizer = tf.optimizers.Adam(learning_rate_schedule)
#     rim.fit(dataset, cost_func, optimizer, 2, max_epochs=50)
#     return rim, dataset
#
#
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     rim, D = test_rim_fit()
#     for (X, Y) in D:
#         break
#     pred, grad = rim.call(X)
#     fig, axs = plt.subplots(2, 5, figsize=(20, 10))
#     k = 0
#     fig.suptitle("Image reconstruction process", fontsize=16)
#     for i in range(2):
#         for j in range(5):
#             if j == 4:
#                 if i == 0:
#                     axs[i, j].imshow(Y.numpy()[0, ..., 0])
#                     axs[i, j].axis("off")
#                     axs[i, j].set_title("Ground Truth")
#                 if i == 1:
#                     axs[i, j].imshow(np.abs(Y.numpy()[0, ..., 0] - pred[..., -1].numpy()[0, ..., 0]))
#                     axs[i, j].axis("off")
#                     axs[i, j].set_title("Residual")
#                 if i == 2:
#                     axs[i, j].axis("off")
#             else:
#                 axs[i, j].imshow(pred[0, ..., 0, k].numpy(), cmap="gray", vmin=-40, vmax=0, origin="lower")
#                 axs[i, j].axis("off")
#                 axs[i, j].set_title(f"Step {k}")
#                 k += 1
#
#     fig, axs = plt.subplots(2, 4, figsize=(20, 10))
#     k = 0
#     fig.suptitle("Gradient", fontsize=16)
#     for i in range(2):
#         for j in range(4):
#             axs[i, j].imshow(grad[0, ..., k].numpy(), cmap="jet", origin="lower")  # , vmin=-1000, vmax=1000)
#             axs[i, j].axis("off")
#             axs[i, j].set_title(f"Step {k}")
#             k += 1
#
#     plt.show()

if __name__ == '__main__':
    test_hparam()