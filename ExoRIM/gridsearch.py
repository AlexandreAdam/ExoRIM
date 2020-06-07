from numpy.random import choice as choose
import numpy as np


def gridsearchV1(model_trained):
    for i in range(model_trained):
        state_depth = choose([2, 4, 8, 16, 32, 64])
        pixels = 32  # Eventually think about more pixels and adjust code for it --> performance challenge
        channels = 1
        steps = choose([3, 6, 9, 12])
        state_size = choose([2, 8])
        reg_amplitude = choose([1e-3, 1e-2, 1e-1])
        noises = choose([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        conv_block_layers = choose([1, 2, 3, 4])
        tconv_block_layers = choose([1, 2, 3, 4])
        downsampling_layers = int((np.log(pixels) - np.log(state_size))/np.log(4))
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
                "filters": conv_block_filters[i+1],
                "kernel_size": [conv_blocks_kernel_size[i+1], conv_blocks_kernel_size[i+1]],
                "strides": [1, 1]
                }
            }
            for j in range(conv_block_layers)
        ]
        tconv_block = [
            {f"TConv_{j+1}": {
                "filters": tconv_block_filters[i+1],
                "kernel_size": [conv_blocks_kernel_size[i+1], conv_blocks_kernel_size[i+1]],
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
                "filters": downsampling_filters,
                "kernel_size": [3, 3],
                "strides": [2, 2]
                }
            }
            for j in range(downsampling_layers)
        ]
        upsampling_block = [
            {f"Conv_Fraction_Stride_{j+1}": {
                "filters": upsampling_filters,
                "kernel_size": [3, 3],
                "strides": [2, 2]
                }
            }
            for j in range(downsampling_layers)
        ]
        yield {
            "grid_id": i,
            "steps": steps,
            "pixels": pixels,
            "channels": channels,
            "state_size": state_size,
            "state_depth": state_depth,
            "Regularizer Amplitude": {
                "kernel": reg_amplitude,
                "bias": reg_amplitude
            },
            "Physical Model": {
                "Visibility Noise": noises,
                "Closure Phase Noise": noises/10
            },
            "Downsampling Block": downsampling_block,
            "Convolution Block": conv_block,
            "Recurrent Block": {
                "GRU_1": {
                    "kernel_size": [gru_kernel_size, gru_kernel_size],
                    "filters": gru_filters
                },
                "Hidden_Conv_1": {
                    "kernel_size": [gru_kernel_size, gru_kernel_size],
                    "filters": gru_filters
                },
                "GRU_2": {
                    "kernel_size": [gru_kernel_size, gru_kernel_size],
                    "filters": gru_filters
                }
            },
            "Upsampling Block": upsampling_block,
            "Transposed Convolution Block": tconv_block
        }