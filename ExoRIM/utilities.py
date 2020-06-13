import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pickle


def save_physical_model_projectors(filename, physical_model):
    arrays = {
        "cos_projector": physical_model.cos_projector,
        "sin_projector": physical_model.sin_projector,
        "bispectra_projector": physical_model.bispectra_projector
    }
    with open(filename, "wb") as f:
        pickle.dump(arrays, f)


def convert_to_8_bit(image):
    return (255.0 / image.max() * (image - image.min())).astype(np.uint8)


def convert_to_float(image):
    "normalize image from uint8 to float32"
    return tf.cast(image, tf.float32)/255.


def save_output(output, dirname, epoch, batch, mod):
    out = output
    if tf.is_tensor(out):
        out = output.numpy()
    if len(out.shape) == 5:
        for instance in range(out.shape[0]):  # over batch size
            image_index = batch*out.shape[0] + instance
            if image_index % mod != 0:
                continue
            for step in range(output.shape[-1]):
                image = convert_to_8_bit(out[instance, :, :, 0, step])
                image = Image.fromarray(image, mode="L")
                image.save(os.path.join(dirname, f"output_{epoch:03}_{image_index:04}_{step:02}.png"))
    elif len(out.shape) == 4:
        for step in range(out.shape[-1]):
            image = convert_to_8_bit(out[:, :, 0, step])
            image = Image.fromarray(image, mode="L")
            image.save(os.path.join(dirname, f"output_{epoch:03}_000_{step:02}.png"))


