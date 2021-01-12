import numpy as np
from ExoRIM.operators import Baselines, redundant_phase_closure_operator, NDFTM, closure_fourier_matrices
from ExoRIM.log_likelihood import bispectrum
from ExoRIM.definitions import rad2mas
import matplotlib.pyplot as plt
from xara.core import phase_binary

pixels = 128
sep = 86  # mas
PA = 102  # degrees
contrast = 5
wavel = 0.5e-6
mask = np.loadtxt("martinache/golay9.txt")
data = np.loadtxt("martinache/gj_164_9_hole_martinache_data.csv", delimiter=",")
print(mask)


def super_gauss0(pixels, ps, sep, PA, w):
    ''' Returns an 2D super-Gaussian function
    ------------------------------------------
    Parameters:
    - (xs, ys) : array size
    - (x0, y0) : center of the Super-Gaussian
    - w        : width of the Super-Gaussian
    ------------------------------------------ '''

    coord = (np.arange(pixels) - pixels//2 + 1/2) * ps
    xx, yy = np.meshgrid(coord, coord)

    x0 = sep * np.cos(np.deg2rad(PA))
    y0 = sep * np.sin(np.deg2rad(PA))

    dist = np.sqrt((xx - x0)**2 + (yy - y0)**2)
    gg = np.exp(-(dist/w)**4)
    return gg


def main():
    # phys = PhysicalModelv1(pixels, mask)
    B = Baselines(mask)
    CPO = redundant_phase_closure_operator(B)

    rho = np.sqrt(B.UVC[:, 0] ** 2 + B.UVC[:, 1] ** 2) / wavel  # frequency in 1/RAD
    theta = rad2mas(1 / rho)  # angular scale covered in mas
    # this picks out the bulk of the baseline frequencies, leaving out poorly constrained lower frequencies
    plate_scale = (np.median(theta) + 6 * np.std(theta)) / pixels
    print(plate_scale * pixels)

    A = NDFTM(B.UVC, wavel, pixels, plate_scale)
    A1, A2, A3 = closure_fourier_matrices(A, CPO)


    image = np.zeros((pixels, pixels))
    ps = plate_scale
    coord = (np.arange(pixels) - pixels//2 + 1/2) * ps


    print(np.deg2rad(PA) / np.pi)
    print(theta)

    image += super_gauss0(pixels, ps, sep/2, PA, 5) / contrast
    image += super_gauss0(pixels, ps, -sep/2, PA, 5)

    bis = bispectrum(image.reshape((1, -1)), A1, A2, A3).numpy().reshape((-1))
    cp = np.angle(bis)

    phase_xara = phase_binary(B.UVC[:, 1], B.UVC[:, 0], wavel, [sep, PA, contrast], deg=False)
    cl_xara = np.rad2deg(CPO @ phase_xara)

    extent = [coord[0] - ps, coord[-1] + ps, coord[0] - ps, coord[-1] + ps]
    fig, ax1 = plt.subplots()
    ax1.imshow(image, extent=extent, cmap="gray", origin="lower")
    ax1.set_ylabel(r"$\theta_y$ [mas]")
    ax1.set_xlabel(r"$\theta_x$ [mas]")
    ax1.set_title(r"sep=86 mas, PA=102 deg, c=5:1")

    plt.savefig("martinache/image.png")
    np.savetxt("martinache/image.txt", image)

    plt.figure()
    plt.plot(np.arange(cp.size), np.rad2deg(cp), "k-", label="exorim", lw=1)
    # plt.plot(data[:, 0], data[:, 1], "r-", label="Martinache2009", lw=1)
    plt.plot(np.arange(cl_xara.size), cl_xara, "b-", label="Xara", lw=1)
    plt.xlabel("Closure triangle")
    plt.ylabel("Closure phase (degrees)")
    plt.legend()
    plt.savefig("martinache/comparison.png")

    np.savetxt("martinache/exorim_closure_phases.txt", cp, header=f"sep={sep} mas, PA={PA} deg, contrast={contrast}:1, pixels={pixels}, "
                                                                  f"plate_scale={plate_scale:5f} mas")
    # ax2.


    plt.show()
    # uv coverage

    # cp = # closure phases


if __name__ == "__main__":
    main()