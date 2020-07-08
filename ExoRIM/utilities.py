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
    return (255.0 * image).astype(np.uint8) # since image is passed through a sigmoid, min and max and force to be 0 and 1


def convert_to_float(image):
    "normalize image from uint8 to float32"
    return tf.cast(image, tf.float32)/255.


def save_output(output, dirname, epoch, batch, index_mod, epoch_mod, step_mod, format="png"):
    if epoch % epoch_mod != 0:
        return
    out = output
    if tf.is_tensor(out):
        out = output.numpy()
    if len(out.shape) == 5:
        # parallelize search for the image
        image_index = np.arange(out.shape[0]) + batch * out.shape[0]
        image_index = image_index[image_index % index_mod == 0]
        step = np.arange(out.shape[-1])
        step = step[step % step_mod == 0]
        step = np.tile(step, reps=[image_index.size, 1])  # fancy broadcasting of the indices
        image_index = np.tile(image_index, reps=[step.shape[1], 1])
        for i, I in enumerate(out[image_index.T, ..., step]): # note that array is reshaped to [batch, steps, pix, pix, channel]
            for j, image in enumerate(I[..., 0]):
                if format == "png":
                    image = convert_to_8_bit(image)
                    image = Image.fromarray(image, mode="L")
                    image.save(os.path.join(dirname, f"output_{epoch:04}_{image_index[i, j]:04}_{step[i, j]:02}.png"))
                elif format == "txt":
                    np.savetxt(os.path.join(dirname, f"output_{epoch:04}_{image_index[i, j]:04}_{step[i, j]:02}.txt"), image)
    elif len(out.shape) == 4:
        # TODO parallelize this one
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
