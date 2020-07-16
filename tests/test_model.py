# from ExoRIM.model import RIM, MSE
# import tensorflow as tf
# import numpy as np
# from datetime import datetime
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from celluloid import Camera
# import os
#
# date_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
#
#
# def simple_image(xp=0, yp=0, sigma=2, pixels=32):
#     image_coords = np.arange(pixels) - pixels / 2.
#     xx, yy = np.meshgrid(image_coords, image_coords)
#     image = np.zeros_like(xx)
#     rho_squared = (xx - xp) ** 2 + (yy - yp) ** 2
#     image += 1 / (sigma * np.sqrt(2. * np.pi)) * np.exp(-0.5 * (rho_squared / sigma**2))
#     image = (image - image.min()) / (image.max() - image.min())
#     return image
#
#
# def test_model_call_and_gradients():
#     pixels = 32
#     noise = 0.01
#
#     # First test call execution
#     rim = RIM(steps=10, pixels=pixels, state_size=4, state_depth=2, noise_std=noise, hidden_cell_features=2)
#     phys = rim.physical_model
#     image = tf.ones((1, pixels, pixels, 1))  # x
#     noisy_image = phys.simulate_noisy_image(image)  # y
#     rim(noisy_image)
#
#     # Test training
#     loss = MSE()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#
#     # train a single epoch
#     with tf.GradientTape() as tape:
#         tape.watch(rim.variables)
#         output = rim.call(noisy_image)
#         cost_value = loss.call(x_true=image, x_preds=output)
#     weight_grads = tape.gradient(cost_value, rim.variables)
#     clipped_grads = [tf.clip_by_value(grads_i, -10, 10) for grads_i in weight_grads]  # normalize weights in [10, 10]
#     optimizer.apply_gradients(zip(clipped_grads, rim.variables))
#
#
# def test_model_learning_on_trivial_image():
#     # We set up a single gaussian at the center, and we test if the model can learn to reconstruct this
#     pixels = 32
#     noise = 1e-4
#     steps = 10
#     epochs = 300
#     step_trace = 10
#
#     test_dir = os.path.join(image_dir, f"learning_test_{date_time}")
#     os.mkdir(test_dir)
#
#     rim = RIM(steps=steps, pixels=pixels, state_size=4, state_depth=2, noise_std=noise, hidden_cell_features=2)
#     image = tf.convert_to_tensor(simple_image().reshape((1, pixels, pixels, 1)), dtype=tf.float32) # x
#     noisy_image = rim.physical_model.simulate_noisy_image(image)  # y
#     fig = plt.figure()
#     plot = plt.imshow(simple_image(), cmap="gray")
#     fig.colorbar(plot)
#     path = os.path.join(test_dir, f"ground_truth.png")
#     fig.savefig(path)
#
#     loss = MSE()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#
#     # train a bunch of epoch
#
#     losses = []
#     outputs = []
#     for epoch in range(epochs):
#         with tf.GradientTape() as tape:
#             tape.watch(rim.trainable_weights)
#             output = rim(noisy_image)
#             cost_value = loss(image, output)
#             cost_value += 0.1 * tf.reduce_sum(tf.square(rim.losses)) # add L2 regularisation to loss
#             outputs.append(output)
#             losses.append(cost_value)
#         weight_grads = tape.gradient(cost_value, rim.trainable_weights)
#         clipped_grads = [tf.clip_by_value(grads_i, -10, 10) for grads_i in weight_grads]  # normalize weights in [10, 10]
#         optimizer.apply_gradients(zip(clipped_grads, rim.trainable_weights))
#
#     # plt.figure()
#     # plt.title("Test with centered gaussian blob")
#     # epoch = np.linspace(0, epochs, len(losses))
#     # plt.plot(epoch, losses, "k-", label="Loss")
#     # plt.xlabel("Epochs")
#     # plt.ylabel("Loss")
#     # plt.legend()
#     # plt.savefig(os.path.join(lossdir, f"Test_loss_{date_time}.png"))
#
#     fig = plt.figure(figsize=(10, 10))
#     camera = Camera(fig)
#     widths = [10, 10, 1]
#     gs = gridspec.GridSpec(2, 3, width_ratios=widths)
#     ax1 = plt.subplot(gs[0, 0])
#     ax2 = plt.subplot(gs[0, 1])
#     ax3 = plt.subplot(gs[1, :2])
#     cbar_ax = plt.subplot(gs[0, 2])
#     cbar_ax.set_ylabel("Intensity", rotation=270)
#     ax3.set_xlabel("Epochs")
#     ax3.set_ylabel("Loss")
#     fig.suptitle("Learn identity with instance norm of grad and L2 regularisation")
#
#     for epoch, output in enumerate(outputs):
#         for step in range(steps):
#             if (step + 1) % step_trace == 0:
#                 image = ax2.imshow(simple_image(), cmap="gray")
#                 cbar = fig.colorbar(image, cax=cbar_ax)
#                 # cbar.ax.get_yaxis().labelpad = 15
#                 # cbar.ax.set_ylabel('Intensity', rotation=270)
#                 ax1.imshow(output[0, :, :, 0, step], cmap="gray")
#                 ax3.plot(np.linspace(0, epoch, len(losses[:epoch])), losses[:epoch], "k-", label="Loss")
#                 camera.snap()
#                 # TODO add the y that the model sees
#                 # path = os.path.join(test_dir, f"epoch_{epoch}_step_{step}.png")
#                 # fig.savefig(path)
#                 # plt.close('all')
#     # ffmpeg -i input.mov -r 0.25 output_%04d.png
#     animation = camera.animate()
#     animation.save(os.path.join(test_dir, "learning_animation.mp4"), writer="ffmpeg")
#
#
# if __name__ == "__main__":
#     test_model_learning_on_trivial_image()

