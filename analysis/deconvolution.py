"""
=============================================
Title: deconvolution

Author(s): Alexandre Adam

Last modified: January 18, 2021

Description: Investigation of classic deconvolution algorithm on 
    known radio data.
=============================================
"""
import numpy as np
import matplotlib.pyplot as plt
import ExoRIM as exo
from ExoRIM.operators import Baselines

N = 8
antenna_positions = np.random.lognormal(mean=0., sigma=3., size=N)

def rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def simulate_uv_coverage()
    """
    For optical data, we usually have 3 or 4 rotations of the mask.
    """
    B = Baselines(antenna_positions)
    for i in range(1, 4):


def main():
    pass

if __name__ == "__main__":
    main()
