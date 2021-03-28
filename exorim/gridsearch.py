from numpy.random import choice as choose
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf


def kfold_splits(X, Y, n, batch_size):
    """

    :param X: numpy arrays of input data
    :param Y: numpy array of ground truth for the input data
    :param n: Number of splits
    :return: a train_dataset and a test_dataset the way the RIM.fit method expect them
    """
    for train_index, test_index in KFold(n_splits=n).split(X, Y):
        X_train, X_test = tf.data.Dataset.from_tensor_slices(X.numpy()[train_index, ...]), tf.data.Dataset.from_tensor_slices(X.numpy()[test_index, ...])
        Y_train, Y_test = tf.data.Dataset.from_tensor_slices(Y.numpy()[train_index, ...]), tf.data.Dataset.from_tensor_slices(Y.numpy()[test_index, ...])
        train_dataset = tf.data.Dataset.zip((X_train, Y_train))
        test_dataset = tf.data.Dataset.zip((X_test, Y_test)).batch(X.numpy()[test_index, ...].shape[0]).cache().prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True).enumerate(start=0).cache().prefetch(tf.data.experimental.AUTOTUNE)
        yield train_dataset, test_dataset


def hparams_for_gridsearchV1(model_trained):
    for i in range(model_trained):
        holes = choose([3, 6, 10, 20, 30, 40]) # those were precomputed in preprocessing
        state_depth = choose([2, 4, 8, 16, 32, 64])
        pixels = 32  # Eventually think about more pixels and adjust code for it --> performance challenge
        channels = 1
        steps = choose([3, 6, 9, 12])
        state_size = choose([4, 8, 16])
        reg_amplitude = choose([1e-3, 1e-2, 1e-1])
        noises = choose([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        conv_block_layers = choose([1, 2, 3, 4])
        tconv_block_layers = choose([1, 2, 3, 4])
        downsampling_layers = int((np.log(pixels) - np.log(state_size))/np.log(2))
        conv_blocks_kernel_size = {
            1: choose([3, 5, 7, 9]),  # possibilities that focus on larger features on first layers
            2: choose([3, 5]),        # same thing for transposed conv block
            3: choose([1, 3]),
            4: choose([1, 3])
        }
        conv_block_filters = {
            1: choose([1, 4]),
            2: choose([4, 8]),
            3: choose([8, 16]),
            4: state_depth//2  # must match hidden tensor dimensions
        }
        # last layer of tconv must have same dimension as the image
        tconv_block_filters = {
            1: choose([8, 16, 32]) if 1 != tconv_block_layers else channels,
            2: choose([8, 16]) if 2 != tconv_block_layers else channels,
            3: choose([4, 8]) if 3 != tconv_block_layers else channels,
            4: channels
        }
        conv_block = [
            {f"Conv_{j+1}": {
                "filters": int(conv_block_filters[j+1]),
                "kernel_size": [int(conv_blocks_kernel_size[j+1]), int(conv_blocks_kernel_size[j+1])],
                "strides": [1, 1]
                }
            }
            for j in range(conv_block_layers)
        ]
        tconv_block = [
            {f"TConv_{j+1}": {
                "filters": int(tconv_block_filters[j+1]),
                "kernel_size": [int(conv_blocks_kernel_size[j+1]), int(conv_blocks_kernel_size[j+1])],
                "strides": [1, 1]
                }
            }
            for j in range(tconv_block_layers)
        ]
        gru_filters = state_depth//2
        gru_kernel_size = choose([3, 5])
        upsampling_filters = gru_filters  # Block connect with recurrent block
        downsampling_filters = channels  # Layer focus on learning to downsample, features are learned in conv block
        downsampling_block = [
            {f"Conv_Downsample_{j+1}": {
                "filters": int(downsampling_filters),
                "kernel_size": [3, 3],
                "strides": [2, 2]
                }
            }
            for j in range(downsampling_layers)
        ]
        upsampling_block = [
            {f"Conv_Fraction_Stride_{j+1}": {
                "filters": int(upsampling_filters),
                "kernel_size": [3, 3],
                "strides": [2, 2]
                }
            }
            for j in range(downsampling_layers)
        ]
        yield {
            "mask_holes": int(holes),
            "grid_id": int(i),
            "steps": int(steps),
            "pixels": int(pixels),
            "channels": int(channels),
            "state_size": int(state_size),
            "state_depth": int(state_depth),
            "Regularizer Amplitude": {
                "kernel": float(reg_amplitude),
                "bias": float(reg_amplitude)
            },
            "Physical Model": {
                "Visibility Noise": float(noises),
                "Closure Phase Noise": float(noises/10.)
            },
            "Downsampling Block": downsampling_block,
            "Convolution Block": conv_block,
            "Recurrent Block": {
                "GRU_1": {
                    "kernel_size": [int(gru_kernel_size), int(gru_kernel_size)],
                    "filters": int(gru_filters)
                },
                "Hidden_Conv_1": {
                    "kernel_size": [int(gru_kernel_size), int(gru_kernel_size)],
                    "filters": int(gru_filters)
                },
                "GRU_2": {
                    "kernel_size": [int(gru_kernel_size), int(gru_kernel_size)],
                    "filters": int(gru_filters)
                }
            },
            "Upsampling Block": upsampling_block,
            "Transposed Convolution Block": tconv_block
        }
