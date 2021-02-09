"""
=============================================
Title: CLEAN

Author(s): 

Last modified: 

Description: A standalone implementation of CLEAN
=============================================
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from astropy.io import fits
from pprint import pprint
import os

# fits index
u_coord_i = 10
v_coord_i = 11
vis_amp_i = 4
vis_phi_i = 6

def polar_to_cartesian(vis, dtype_complex=True):
    r = np.sqrt(vis[:, 0]**2 + vis[:, 1]**2)
    theta = np.arctan2(vis[:, 1], vis[:, 0])
    if dtype_complex:
        return r * np.cos(theta) + 1j* r * np.sin(theta)
    else:
        return np.array([r * np.cos(theta), r * np.sin(theta)]).T

def clean(vis, uvc):

def main(datapath, image):
    fit = fits.open(os.path.join(datapath, image))
    wavel = fit[1].data[0][0] # works with vlbi oifits files
    data = fit[4].data
    data = list(zip(*data)) # transpose the table
    uvc = np.array([data[u_coord_i], data[v_coord_i]]).T # uv coordinates, in meter
    vis = np.array([data[vis_amp_i], np.deg2rad(data[vis_phi_i])]).T # visibility in polar form
    vis = polar_to_cartesian(vis)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--datapath", default= os.path.expandvars("$project/data/UVdata"), type=str, help="Path to UVdata directory")
    parser.add_argument("-i", "--image", type=str, default="natural-01-20.oifits", help="Name of the dataset and image")
    args = parser.parse_args()
    main(args.datapath, args.image)
