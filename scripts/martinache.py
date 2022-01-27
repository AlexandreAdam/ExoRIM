import numpy as np
from exorim.operators import Baselines, redundant_phase_closure_operator, NDFTM, closure_fourier_matrices, closure_baselines_projectors
from exorim.inference import bispectrum
from exorim.definitions import rad2mas, mas2rad
import matplotlib.pyplot as plt
from xara.core import cvis_binary

pixels = 248
sep = 86  # mas
PA = 102  # degrees
contrast = 5
wavel = 1.7e-6 # hband, see Keck, Palomar

# Splodges -- overlay fft of mask and 

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


def general_gamma_binary(uv):
    """
    Complex visibility for 2 delta function
    """
    x, y  = uv[:, 0], uv[:, 1]
    k = 2 * np.pi / wavel
    beta = mas2rad(sep)
    th = np.deg2rad(PA)
    i2 = 1
    i1 = 1 / contrast
    phi1 = k * x * beta * np.cos(th)/2
    phi2 = k * y * beta * np.sin(th)/2
    out = i1 * np.exp(-1j * (phi1 + phi2))
    out += i2 * np.exp(1j * (phi1 + phi2))
    return out/ (i1 + i2)


def main():
    # phys = PhysicalModelv1(pixels, mask)
    B = Baselines(mask)
    CPO = redundant_phase_closure_operator(B)

    rho = np.sqrt(B.UVC[:, 0] ** 2 + B.UVC[:, 1] ** 2) / wavel  # frequency in 1/RAD
    theta = rad2mas(1 / rho)  # angular scale covered in mas
    # this picks out the bulk of the baseline frequencies, leaving out poorly constrained lower frequencies
    plate_scale = (np.median(theta) + 6 * np.std(theta)) / pixels
    A = NDFTM(B.UVC, wavel, pixels, plate_scale) # Direct Fourier Transform Matrix
    A1, A2, A3 = closure_fourier_matrices(A, CPO)
    V1, V2, V3 = closure_baselines_projectors(CPO)

    # Create an image
    image = np.zeros((pixels, pixels))
    ps = plate_scale
    coord = (np.arange(pixels) - pixels//2 + 1/2) * ps
    image += super_gauss0(pixels, ps, sep/2, PA, 5) / contrast
    image += super_gauss0(pixels, ps, -sep/2, PA, 5)

    # Compute the bispectrum and closure phase
    phase = np.angle(A @ image.flatten())
    bis = bispectrum(image.reshape((1, -1)), A1, A2, A3).numpy().reshape((-1))
    cp = np.angle(bis)

    # Frantz Martinache code for a binary complex visibility
    gamma_xara = cvis_binary(B.UVC[:, 0], B.UVC[:, 1], wavel, [sep, PA, contrast])
    phase_xara = np.angle(gamma_xara)
    gx1 = V1 @ gamma_xara
    gx2 = V2 @ gamma_xara
    gx3 = V3 @ gamma_xara
    bis_xara = gx1 * gx2.conj() * gx3
    cp_xara = np.angle(bis_xara)


    # Mathematical model for 2 delta functions, should be same thing as Martinache code
    gamma_model = general_gamma_binary(B.UVC)
    phase_model = np.angle(gamma_model)
    g1 = V1 @ gamma_model
    g2 = V2 @ gamma_model
    g3 = V3 @ gamma_model
    bis_model = g1 * g2.conj() * g3
    cp_model = np.rad2deg(np.angle(bis_model))

    # Compare
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(phase), "k-", label="exorim", lw=1)
    # ax.plot(np.rad2deg(phase_xara), "b-", label="Xara", lw=1)
    ax.plot(np.rad2deg(phase_model), "g-", label="expected", lw=1)
    ax.set_title("Binary Phase comparison")
    ax.set_ylabel(r"$\varphi$")
    ax.legend()
    plt.savefig("martinache/phase_comparison.png")

    extent = [coord[0] - ps, coord[-1] + ps, coord[0] - ps, coord[-1] + ps]
    fig, ax1 = plt.subplots()
    ax1.imshow(image, extent=extent, cmap="gray", origin="lower")
    ax1.set_ylabel(r"$\theta_y$ [mas]")
    ax1.set_xlabel(r"$\theta_x$ [mas]")
    ax1.set_title(r"sep=86 mas, PA=102 deg, c=5:1")

    # plt.savefig("martinache/image.png")
    # np.savetxt("martinache/image.txt", image)

    plt.figure()
    plt.plot(np.sort(np.rad2deg(cp)), "k-", label="exorim", lw=1)
    plt.plot(np.sort(data[:, 1]), "r-", label="Martinache2009", lw=1)
    plt.plot(np.sort(np.rad2deg(cp_xara)), "b-", label="Xara", lw=1)
    plt.plot(np.sort(cp_model), "g-", label="expected", lw=1)
    plt.xlabel("Closure triangle")
    plt.ylabel("Closure phase (degrees)")
    plt.legend()
    # plt.savefig("martinache/comparison.png")

    # np.savetxt("martinache/exorim_closure_phases.txt", cp, header=f"sep={sep} mas, PA={PA} deg, contrast={contrast}:1, pixels={pixels}, "
                                                                  # f"plate_scale={plate_scale:5f} mas")
    # ax2.


    plt.show()
    # uv coverage

    # cp = # closure phases


if __name__ == "__main__":
    main()
