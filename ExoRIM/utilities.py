import numpy as np
import tensorflow as tf
from PIL import Image
import os, glob
import pickle

AUTOTUNE = tf.data.experimental.AUTOTUNE

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


def save_output(output, dirname, epoch, batch, index_mod, epoch_mod, step_mod):
    if epoch % epoch_mod != 0:
        return
    out = output
    if tf.is_tensor(out):
        out = output.numpy()
    if len(out.shape) == 5:
        for instance in range(out.shape[0]):  # over batch size
            image_index = batch*out.shape[0] + instance
            if image_index % index_mod != 0:
                continue
            for step in range(output.shape[-1]):
                if step % step_mod != 0:
                    continue
                image = convert_to_8_bit(out[instance, :, :, 0, step])
                image = Image.fromarray(image, mode="L")
                image.save(os.path.join(dirname, f"output_{epoch:03}_{image_index:04}_{step:02}.png"))
    elif len(out.shape) == 4:
        for step in range(out.shape[-1]):
            if step % step_mod == 0:
                continue
            image = convert_to_8_bit(out[:, :, 0, step])
            image = Image.fromarray(image, mode="L")
            image.save(os.path.join(dirname, f"output_{epoch:03}_000_{step:02}.png"))


# def create_datasets(meta_data, rim, dirname, batch_size=None):
#     images = tf.convert_to_tensor(create_and_save_data(dirname, meta_data), dtype=tf.float32)
#     k_images = rim.physical_model.simulate_noisy_image(images)
#     X = tf.data.Dataset.from_tensor_slices(k_images)  # split along batch dimension
#     Y = tf.data.Dataset.from_tensor_slices(images)
#     dataset = tf.data.Dataset.zip((X, Y))
#     if batch_size is not None: # for train set
#         dataset = dataset.batch(batch_size, drop_remainder=True)
#         dataset = dataset.enumerate(start=0)
#         dataset = dataset.cache()  # accelerate the second and subsequent iterations over the dataset
#         dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs
#     else:
#         # batch together all examples, for test set
#         dataset = dataset.batch(images.shape[0], drop_remainder=True)
#         dataset = dataset.cache()
#     return dataset


def load_dataset(dirname, rim, batch_size=None):
    images = []
    for file in glob.glob(os.path.join(dirname, "*.png")):
        with Image.open(file) as image:
            im = np.array(image.getdata()).reshape([1, image.size[0], image.size[1], 1])
            images.append(im)
    images = tf.convert_to_tensor(np.concatenate(images, axis=0))
    images = convert_to_float(images)
    k_images = rim.physical_model.simulate_noisy_image(images)
    X = tf.data.Dataset.from_tensor_slices(k_images)
    Y = tf.data.Dataset.from_tensor_slices(images)
    dataset = tf.data.Dataset.zip((X, Y))
    if batch_size is not None: # for train set
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.enumerate(start=0)
        dataset = dataset.cache()  # accelerate the second and subsequent iterations over the dataset
        dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs
    else:
        # batch together all examples, for test set
        dataset = dataset.batch(images.shape[0], drop_remainder=True)
        dataset = dataset.cache()
    return dataset
