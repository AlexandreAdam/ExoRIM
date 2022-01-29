import numpy as np
import tensorflow as tf
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import CenteredNorm
import io

try:
    from contextlib import nullcontext  # python > 3.7 needed for this
except ImportError:
    # Backward compatibility with python <= 3.6
    class nullcontext:
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


class nullwriter:
    @staticmethod
    def flush():
        pass

    @staticmethod
    def as_default():
        return nullcontext()


class nulltape(nullcontext):
    @staticmethod
    def stop_recording():
        return nullcontext()

    @staticmethod
    def flush():
        pass


def convert_to_8_bit(image):
    return (255.0 * image).astype(np.uint8)


def convert_to_float(image):
    "normalize image from uint8 to float32"
    return tf.cast(image, tf.float32)/255.


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def residual_plot(dataset, rim, N):
    fig, axs = plt.subplots(N, 5, figsize=(6 * 4, N * 4))
    index = [0, 1, -1]
    label = [1, 2, rim.steps]
    for j in range(N):
        X, Y, noise_std = dataset[j]
        y = Y[0, ..., 0]
        out, chi_squared = rim.call(X, noise_std)
        out = rim.link_function(out[:, 0, ..., 0])
        for plot_i, i in enumerate(index):
            axs[j, plot_i].imshow(out[i], cmap="bone", origin="lower", vmin=0, vmax=1)
            axs[j, plot_i].axis("off")
            if j == 0:
                axs[j, plot_i].set_title(fr"Step {label[i]}: $\chi^2_\nu = {chi_squared[i, 0]:.1f}$")

        axs[j, 3].imshow(y, cmap="bone", origin="lower", vmin=0, vmax=1)
        axs[j, 3].axis("off")

        im = axs[j, 4].imshow(out[-1] - y, cmap="seismic", origin="lower")
        divider = make_axes_locatable(axs[j, 4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[j, 4].axis("off")
        if j == 0:
            axs[j, 3].set_title(f"Ground Truth")
            axs[j, 4].set_title(f"Residuals")
    plt.subplots_adjust(wspace=.4, hspace=.2)
    return fig


if __name__ == '__main__':
    from exorim.simulated_data import CenteredBinariesDataset
    from exorim import RIM, PhysicalModel, Model

    phys = PhysicalModel(pixels=32)
    dataset = CenteredBinariesDataset(phys, total_items=10, batch_size=1)
    model = Model()
    rim = RIM(model, phys, time_steps=3)
    N = 2
    residual_plot(dataset, rim, N)
    plt.show()
