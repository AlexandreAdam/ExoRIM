from ExoRIM.model import RIM, MSE, PhysicalModel
import tensorflow as tf


def test_model():
    pixels = 32
    noise = 0.01
    phys = PhysicalModel(pixels=pixels, noise_std=noise)
    image = tf.ones((1, pixels, pixels, 1))  # x
    noisy_image = phys.simulate_noisy_image(image)  # y

    # First test call execution
    rim = RIM(steps=10, pixels=32, state_size=4, state_depth=2, noise_std=0.1, num_cell_features=2)
    rim(noisy_image)

    # Test training
    loss = MSE()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # train a single epoch
    with tf.GradientTape() as tape:
        tape.watch(rim.variables)
        output = rim.call(noisy_image)
        cost_value = loss.call(x_true=image, x_preds=output)
    weight_grads = tape.gradient(cost_value, rim.variables)
    clipped_grads = [tf.clip_by_value(grads_i, -10, 10) for grads_i in weight_grads]  # normalize weights in [10, 10]
    optimizer.apply_gradients(zip(clipped_grads, rim.variables))


if __name__ == "__main__":
    test_model()

