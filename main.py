from ExoRIM.trainv2 import Training
import tensorflow as tf
import numpy as np

coords = np.random.randn(10, 2)
np.savetxt("coords.txt", coords)

train = Training(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    model_name="RIM",
    epochs=5,
    total_items=10,
    split=0.8,
    batch_size=2,
    checkpoints=None,
    images_saved=2,
    steps=12,  # number of steps for the reconstruction
    pixels=32,
    state_size=8,  # hidden state 2D size
    state_depth=2,  # Channel dimension of hidden state
    noise_std=0.0001,  # This is relative to the smallest complex visibility
    num_cell_features=2,
    step_trace=[3, 8, 11],  # starting from 0
)
train.train_weights()
train.save_movies("RIM")