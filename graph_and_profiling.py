import tensorflow as tf
import numpy as np
from ExoRIM.physical_model import PhysicalModel
from ExoRIM.definitions import default_hyperparameters, dtype, mycomplex
from ExoRIM import RIM
import os
import json


def main(id):
    logdir = "./logs/" + id
    profiledir = logdir + "/profile"
    try:
        os.mkdir(profiledir)
    except FileExistsError:
        pass
    with open("hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)
    image = tf.constant(np.random.uniform(0, 1, (1, 32, 32, 1)), dtype)  # tf.TensorShape((None, 32, 32, 1))
    mask = np.random.normal(0, 1, (7, 2))
    phys = PhysicalModel(
        pixels=32,
        mask_coordinates=mask,
        wavelength=0.5e-6,
        plate_scale=3.2,
        SNR=100
    )
    rim = RIM(physical_model=phys, hyperparameters=hyperparameters)
    ht = rim.init_hidden_states(1)
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)
    # data = phys.simulate_noisy_data(image)
    rim.model.call(image, ht)
    with writer.as_default():
        tf.summary.trace_export(name="RIM", step=0, profiler_outdir=profiledir)



if __name__ == "__main__":
    main("20-07-28_04-47-23")
