from ExoRIM.model import RIM, MSE
from ExoRIM.data_generator import SimpleGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ExoRIM.definitions import lossdir, modeldir
from datetime import datetime
import os

date_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")


def train():
    epochs = 10
    train_trace = 3  # save a trace of the loss each x batch
    test_trace = 1
    rim = RIM(steps=4, pixels=32, state_size=4, state_depth=2, noise_std=0.1, num_cell_features=2)
    phys = rim.physical_model
    generator = SimpleGenerator(30) # 1000 pictures, 800 in training

    loss = MSE()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    train_losses = []
    train_trace_loss = 0  # these are tracers, reset at trace percent of an epoch
    test_trace_loss = 0
    test_losses = []

    for i in range(epochs):
        # training
        for image in generator.training_batch():
            image = tf.convert_to_tensor(image, dtype=tf.float32)  # x
            noisy_image = phys.simulate_noisy_image(image)  # y
            with tf.GradientTape() as tape:
                tape.watch(rim.variables)
                output = rim(noisy_image)
                cost_value = loss(image, output)
                train_trace_loss += cost_value
            if (generator.train_index + 1) % train_trace == 0:
                train_losses.append(train_trace_loss)
                print(f"epoch {i+1}: training loss = {train_trace_loss}")
                train_trace_loss = 0  # reset the trace
            weight_grads = tape.gradient(cost_value, rim.trainable_weights)
            # prevent exploding gradient
            clipped_grads = [tf.clip_by_value(grads_i, -10, 10) for grads_i in weight_grads]
            optimizer.apply_gradients(zip(clipped_grads, rim.trainable_weights))

        # test
        for image in generator.test_batch():
            image = tf.convert_to_tensor(image, dtype=tf.float32)  # x
            noisy_image = phys.simulate_noisy_image(image)  # y
            output = rim.call(noisy_image)
            cost_value = loss.call(x_true=image, x_preds=output)
            test_trace_loss += cost_value
        if (generator.test_index + 1) % test_trace == 0:
            test_losses.append(test_trace_loss)
            print(f"epoch {i+1}: test loss = {test_trace_loss}")
            test_trace_loss = 0  # reset the trace
    rim.save_weights(os.path.join(modeldir, f"rim_{date_time}.h5"))
    return epochs, train_losses, test_losses


def loss_curve():

    epochs, train_loss, test_loss = train()
    plt.figure()
    # for 10 epochs, #TODO should work for any number of epochs or trace
    # and for 10 tracers per epochs
    trains = np.linspace(0, epochs, len(train_loss))
    test = np.linspace(0, epochs, len(test_loss))
    plt.plot(trains, train_loss, "k-", label="Training loss")
    plt.plot(test, test_loss, "r-", label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(lossdir, f"loss_curve_{date_time}.png"))


if __name__ == "__main__":
    loss_curve()
