from ExoRIM.kpi import kpi
from ExoRIM.definitions import dtype
import tensorflow as tf
import numpy as np
import os
import pickle
from argparse import ArgumentParser


# Saves projector arrays and mask to datadir to be picked up by gridsearch
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--large", action="store_true", help="evaluate bispectra for large "
                                                             "number of holes stored in script. "
                                                             "This might blow up your RAM -- to be used in "
                                                             "cluster")
    args = parser.parse_args()
    basedir = os.getcwd()
    datadir = os.path.join(basedir, "data", "projector_arrays")
    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    x = np.arange(32)  # 32 pixels
    xx, yy = np.meshgrid(x, x)
    holes_list = [3, 6, 10] if args.large is False else [20, 30, 40, 50, 60, 100]
    for holes in holes_list:
        mask_coordinates = np.random.normal(size=(holes, 2))
        bs = kpi(mask=mask_coordinates)
        p2vm_sin = np.zeros((bs.uv.shape[0], xx.ravel().shape[0]))

        # TODO change this for FFT, might be too slow
        for j in range(bs.uv.shape[0]):
            p2vm_sin[j, :] = np.ravel(np.sin(2 * np.pi * (xx * bs.uv[j, 0] + yy * bs.uv[j, 1])))

        p2vm_cos = np.zeros((bs.uv.shape[0], xx.ravel().shape[0]))

        for j in range(bs.uv.shape[0]):
            p2vm_cos[j, :] = np.ravel(np.cos(2 * np.pi * (xx * bs.uv[j, 0] + yy * bs.uv[j, 1])))

        # create tensor to hold cosine and sine projection operators
        cos_projector = tf.constant(p2vm_cos.T, dtype=dtype)
        sin_projector = tf.constant(p2vm_sin.T, dtype=dtype)
        bispectra_projector = tf.constant(bs.uv_to_bsp.T, dtype=dtype)
        with open(os.path.join(datadir, f"projectors_{holes}_holes.pickle"), "wb") as f:
            arrays = {
                "cos_projector": cos_projector,
                "sin_projector": sin_projector,
                "bispectra_projector": bispectra_projector
            }
            pickle.dump(arrays, f)
        np.savetxt(os.path.join(datadir, f"mask_{holes}_holes.txt"), mask_coordinates)