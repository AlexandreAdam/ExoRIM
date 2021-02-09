import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from ExoRIM.definitions import mas2rad, rad2mas

N = 1028
sep = 86 # mas
theta = 102 # degrees
ps = 2 * sep / N # plate scale in mas
wavelength = 0.5e-6 # meter



def binary(xx, yy, sep, i1, i2):
    # put z axis between point stars
    out = i1 * ((xx - sep/2) == 0) * (yy == 0)
    out += i2 * ((xx - sep/2) == 0) * (yy == 0)
    return out 

def gamma_binary(x, y, i1, i2):
    k = 2 * np.pi / wavelength
    beta = mas2rad(sep)
    out = (i1 - i2) / (i1 + i2) * np.exp(1j * k * x * beta / 2)
    out += 2 * i2 / (i1 + i2) * np.cos(np.pi * k * x * beta / 2)
    return out


def general_gamma_binary(x, y, i1, i2):
    k = 2 * np.pi / wavelength
    beta = mas2rad(sep)
    th = np.deg2rad(theta)
    phi1 = k * x * beta * np.cos(th)/2
    phi2 = k * y * beta * np.sin(th)/2
    out = i1 * np.exp(-1j * (phi1 + phi2))
    out += i2 * np.exp(1j * (phi1 + phi2))
    return out/ (i1 + i2)

def imshow(image, ax):
    im = ax.imshow(image, cmap="gray", norm=Normalize(0, 1), origin="lower")
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def main():
    i1 = 1
    i2 = 0.2 
    x = np.linspace(-4, 4, N)
    xx, yy = np.meshgrid(x, x)
    gam = gamma_binary(xx, yy, i1, i2)
    gam = general_gamma_binary(xx, yy, i1, i2)
    fig, ((axv, axr), (axi, axphi)) = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(rf"I1={i1}, I2={i2}, $\beta$={sep} mas")
    imshow(2 * i1 * i2 / (i1**2 + i2**2) * np.abs(gam), axv)
    axv.set_title("Visbility")
    imshow(gam.real, axr)
    axr.set_title(r"$\Re(\gamma)$")
    imshow(gam.imag, axi)
    axi.set_title(r"$\Im(\gamma)$")
    im = axphi.imshow(np.angle(gam), origin="lower")
    axphi.axis("off")
    divider = make_axes_locatable(axphi)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    axphi.set_title(r"$\cos$(arg($\gamma$))")
    print(np.angle(gam))
    plt.show()




if __name__ == "__main__":
    main()


