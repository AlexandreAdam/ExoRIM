from definitions import RIM_CELL
import numpy as np
import matplotlib as mpl
import tensorflow as tf
from kpi import kpi

mpl.rcParams['image.origin'] = 'lower'
mpl.rcParams['figure.figsize'] = (10.0,10.0)    #(6.0,4.0)
mpl.rcParams['font.size'] = 16              #10
mpl.rcParams['savefig.dpi'] = 300             #72
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'

colours = mpl.rcParams['axes.prop_cycle'].by_key()['color']

tf.enable_eager_execution()


if __name__ == "__main__":
    # x = np.arange(128)
    # xx, yy = np.meshgrid(x, x) # Image grid
    # coords = np.random.choice(6, 2) # Interferometric array coordinates
    baseline = kpi(file="coord.txt")


    pass