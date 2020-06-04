import json


def load_hyperparameters(file):
    with open(file, "r") as f:
        return json.load(f)


def empty_hyperparameters_dict():
    return {
        "steps": None,
        "pixels": None,
        "channels": 1,
        "state_size": None,
        "state_depth": None,
        "Regularizer Amplitude": {},
        "Physical Model": {},
        "Input": {},
        "Downsampling Block": [],
        "Convolution Block": [],
        "Recurrent Block": [],
        "Upsampling Block": [],
        "Transposed Convolution Block": []
    }



