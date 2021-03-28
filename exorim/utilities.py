import numpy as np
import tensorflow as tf
from exorim.interferometry.simulated_data import CenteredImagesGenerator
from exorim.interferometry.models.physical_model import PhysicalModel
from exorim.definitions import DTYPE, mycomplex
from PIL import Image
import os, glob
import pickle
import collections
try:
    from contextlib import nullcontext  # python > 3.7 needed for this
except ImportError:
    # Backward compatibility with python <= 3.6
    class nullcontext:
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

AUTOTUNE = tf.data.experimental.AUTOTUNE


class nullwriter:
    @staticmethod
    def flush():
        pass

    @staticmethod
    def as_default():
        return nullcontext()


def save_physical_model_projectors(filename, physical_model):
    arrays = {
        "cos_projector": physical_model.cos_projector,
        "sin_projector": physical_model.sin_projector,
        "bispectra_projector": physical_model.bispectra_projector
    }
    with open(filename, "wb") as f:
        pickle.dump(arrays, f)


def convert_to_8_bit(image):
    return (255.0 * image).astype(np.uint8)


def convert_to_float(image):
    "normalize image from uint8 to float32"
    return tf.cast(image, tf.float32)/255.


def save_output(output, dirname, epoch, batch, index_mod, epoch_mod, timestep_mod, format="png", step_mod=None):
    if epoch % epoch_mod != 0:
        return
    out = output
    if tf.is_tensor(out):
        out = output.numpy()
    if len(out.shape) == 5:
        # parallelize search for the image
        image_index = np.arange(out.shape[0])
        true_image_index = image_index + batch * out.shape[0]
        image_index = image_index[(true_image_index) % index_mod == 0]
        timestep = np.arange(out.shape[-1])
        timestep = timestep[(timestep + 1) % timestep_mod == 0]
        timestep = np.tile(timestep, reps=[image_index.size, 1])  # fancy broadcasting of the indices
        image_index = np.tile(image_index, reps=[timestep.shape[1], 1])
        for i, I in enumerate(out[image_index.T, ..., timestep]): # note that array is reshaped to [batch, steps, pix, pix, channel]
            for j, image in enumerate(I[..., 0]):
                if format == "png":
                    image = convert_to_8_bit(image)
                    image = Image.fromarray(image, mode="L")
                    image.save(os.path.join(dirname, f"output_{epoch:04}_{true_image_index[i]:04}_{timestep[i, j]:02}.png"))
                elif format == "txt":
                    np.savetxt(os.path.join(dirname, f"output_{epoch:04}_{true_image_index[i]:04}_{timestep[i, j]:02}.txt"), image)
    elif len(out.shape) == 4:
        # TODO parallelize this one
        for timestep in range(out.shape[-1]):
            if timestep % timestep_mod == 0:
                continue
            image = convert_to_8_bit(out[:, :, 0, timestep])
            image = Image.fromarray(image, mode="L")
            image.save(os.path.join(dirname, f"output_{epoch:03}_000_{timestep:02}.png"))


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def save_gradient_and_weights(grad, trainable_weight, dirname, epoch, batch):
    file = os.path.join(dirname, "grad_and_weights.pickle")
    if os.path.exists(file):
        with open(file, "rb") as f:
            d = pickle.load(f)
    else:
        d = {}
    for i, _ in enumerate(grad):
        layer = trainable_weight[i].name
        update(d, {layer : {epoch : {batch : {
            "grad_mean": grad[i].numpy().mean(),
            "grad_var" : grad[i].numpy().std(),
            "weight_mean": trainable_weight[i].numpy().mean(),
            "weight_var": trainable_weight[i].numpy().std(),
            "weight_max": trainable_weight[i].numpy().max(),
            "weight_min": trainable_weight[i].numpy().min()
        }}}})
    with open(file, "wb") as f:
        pickle.dump(d, f)


def save_loglikelihood_grad(grad, dirname, epoch, batch, index_mod, epoch_mod, timestep_mod, step_mod):
    if epoch % epoch_mod != 0:
        return
    out = grad.numpy()
    image_index = np.arange(out.shape[0])
    true_image_index = image_index + batch * out.shape[0]
    image_index = image_index[(true_image_index) % index_mod == 0]
    timestep = np.arange(out.shape[-1])
    timestep = timestep[(timestep + 1) % timestep_mod == 0]
    timestep = np.tile(timestep, reps=[image_index.size, 1])
    image_index = np.tile(image_index, reps=[timestep.shape[1], 1])
    for i, G in enumerate(out[image_index.T, ..., timestep]):
        for j, g in enumerate(G):
            np.savetxt(os.path.join(dirname, f"grad_{epoch:04}_{true_image_index[i]:04}_{timestep[i, j]:02}.txt"), g)


# Uses the simulated_data/CenteredImagesGenerator class
def create_dataset_from_generator(
        physical_model: PhysicalModel, #TODO make an abstract class to avoid circular import
        item_per_epoch=1000,
        batch_size=50,
        highest_contrast=0.5,
        max_point_source=10,
        fixed=False,
        seed=None
):
    gen = CenteredImagesGenerator(
        physical_model=physical_model,
        total_items_per_epoch=item_per_epoch,
        channels=1,
        highest_contrast=highest_contrast,
        max_point_sources=max_point_source,
        fixed=fixed
    )
    if fixed and seed is not None:
        gen.epoch = seed
    pixels = physical_model.pixels
    shapes = (tf.TensorShape([None]), tf.TensorShape([pixels, pixels, 1]))
    dataset = tf.data.Dataset.from_generator(gen.generator, output_types=(mycomplex, DTYPE), output_shapes=shapes)
    dataset = dataset.cache()              # accelerate the second and subsequent iterations over the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # dataset = dataset.enumerate(start=0)
    dataset = dataset.prefetch(AUTOTUNE)  # Batch is prefetched by CPU while training on the previous batch occurs
    return dataset


def replay_dataset_from_generator(dataset:tf.data.Dataset, epochs, dirname, format, fixed, index_mod=1, epoch_mod=1):
    if fixed:
        max_epoch = 1
    else:
        max_epoch = epochs
    for epoch in range(max_epoch):
        image_index = 0
        if epoch % epoch_mod == 0:
            batch = -1
            for (vis, image) in dataset:
                batch += 1
                for im in image.numpy():  # iterate over batch
                    if image_index % index_mod != 0:
                        image_index += 1
                        continue
                    im = im[..., 0]  # remove channel dim
                    if format == "png":
                        im = convert_to_8_bit(im)
                        im = Image.fromarray(im, mode="L")
                        im.save(os.path.join(dirname, f"image_{epoch:04}_{image_index:04}.png"))
                    elif format == "txt":
                        np.savetxt(os.path.join(dirname, f"image_{epoch:04}_{image_index:04}.txt"), im)
                    image_index += 1

# TODO eventually make a dataset with random uv coverage


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



# #TODO work on these functions
# # from collections import defaultdict, namedtuple
# # from typing import List
# # import tensorflow as tf
#
#
# def extract_images_from_event(event_filename: str, image_tags: List[str]):
#     from tensorflow.python.summary.summary_iterator import summary_iterator
#
#     TensorBoardImage = namedtuple("TensorBoardImage", ["topic", "image", "cnt"])
#     topic_counter = defaultdict(lambda: 0)
#     serialized_examples = tf.data.TFRecordDataset(event_filename)
#     for serialized_example in serialized_examples:
#         event = tf.Event.FromString(serialized_example.numpy())
#         for v in summary_iterator("/path/to/log/file"):
#             if v.tag in image_tags:
#                 if v.HasField('tensor'):  # event for images using tensor field
#                     s = v.tensor.string_val[2]  # first elements are W and H
#
#                     tf_img = tf.image.decode_image(s)  # [H, W, C]
#                     np_img = tf_img.numpy()
#
#                     topic_counter[v.tag] += 1
#
#                     cnt = topic_counter[v.tag]
#                     tbi = TensorBoardImage(topic=v.tag, image=np_img, cnt=cnt)
#
#                     yield tbi
#
#
#
#
# def tabulate_events(dpath):
#     from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
#     summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
#
#     tags = summary_iterators[0].Tags()['scalars']
#
#     for it in summary_iterators:
#         assert it.Tags()['scalars'] == tags
#
#     out = defaultdict(list)
#     steps = []
#
#     for tag in tags:
#         steps = [e.step for e in summary_iterators[0].Scalars(tag)]
#
#         for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
#             assert len(set(e.step for e in events)) == 1
#
#             out[tag].append([e.value for e in events])
#
#     return out, steps
#
#
# def to_csv(dpath):
#     import pandas as pd
#     dirs = os.listdir(dpath)
#
#     d, steps = tabulate_events(dpath)
#     tags, values = zip(*d.items())
#     np_values = np.array(values)
#
#     for index, tag in enumerate(tags):
#         df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
#         df.to_csv(get_file_path(dpath, tag))
#
#
# def get_file_path(dpath, tag):
#     file_name = tag.replace("/", "_") + '.csv'
#     folder_path = os.path.join(dpath, 'csv')
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     return os.path.join(folder_path, file_name)