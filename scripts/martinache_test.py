import numpy as np
from exorim import PhysicalModel
from exorim.definitions import rad2mas, general_gamma_binary, super_gaussian
import matplotlib.pyplot as plt
from xara.core import cvis_binary

pixels = 248
sep = 86  # mas
PA = 102  # degrees
contrast = 5
wavel = 1.7e-6 # hband, see Keck, Palomar

mask = np.loadtxt("martinache/golay9.txt")
data = np.loadtxt("martinache/gj_164_9_hole_martinache_data.csv", delimiter=",")
print(mask)



def main():
    phys = PhysicalModel(pixels, mask, wavel, oversampling_factor=None)
    B = phys.operators
    CPO = B.CPO
    A1, A2, A3 = phys.A1.numpy(), phys.A2.numpy(), phys.A3.numpy()
    V1, V2, V3 = phys.V1.numpy(), phys.V2.numpy(), phys.V3.numpy()
    A = phys.A.numpy()

    plate_scale = phys.plate_scale # (np.median(theta) + 6 * np.std(theta)) / pixels
    image = np.zeros((pixels, pixels))
    ps = plate_scale
    coord = (np.arange(pixels) - pixels//2 + 1/2) * ps
    image += super_gaussian(pixels, ps, sep/2, PA, 5) / contrast
    image += super_gaussian(pixels, ps, -sep/2, PA, 5)

    # Compute the bispectrum and closure phase using our Physical Model
    phase = np.angle(phys.visibility(image[None, ..., None]))[0]
    bis = phys.bispectrum(image[None, ..., None])[0]
    cp = np.angle(bis)

    # Frantz Martinache code for a binary complex visibility
    gamma_xara = cvis_binary(B.UVC[:, 0], B.UVC[:, 1], wavel, [sep, PA, contrast])
    phase_xara = np.angle(gamma_xara)
    gx1 = V1 @ gamma_xara
    gx2 = V2 @ gamma_xara
    gx3 = V3 @ gamma_xara
    bis_xara = gx1 * gx2.conj() * gx3
    cp_xara = np.angle(bis_xara)

    # Mathematical model for 2 delta functions
    gamma_model = general_gamma_binary(B.UVC, wavel, sep, PA, contrast)
    phase_model = np.angle(gamma_model)
    g1 = V1 @ gamma_model
    g2 = V2 @ gamma_model
    g3 = V3 @ gamma_model
    bis_model = g1 * g2.conj() * g3
    cp_model = np.rad2deg(np.angle(bis_model))

    # Compare
    fig, ax = plt.subplots()
    ax.plot(np.sort(np.rad2deg(phase)), "k-", label="exorim", lw=1)
    ax.plot(np.sort(np.rad2deg(phase_xara)), "b-", label="Xara", lw=1)
    ax.plot(np.sort(np.rad2deg(phase_model)), "g-", label="expected", lw=1)
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

    plt.savefig("martinache/image.png")
    # np.savetxt("martinache/image.txt", image)

    plt.figure()
    plt.plot(np.sort(np.rad2deg(cp)), "k-", label="exorim", lw=1)
    # plt.plot(np.sort(data[:, 1]), "r-", label="Martinache2009", lw=1)
    plt.plot(np.sort(np.rad2deg(cp_xara)), "b-", label="Xara", lw=1)
    plt.plot(np.sort(cp_model), "g-", label="expected", lw=1)
    plt.xlabel("Closure triangle")
    plt.ylabel("Closure phase (degrees)")
    plt.legend()
    plt.savefig("martinache/comparison.png")

    np.savetxt("martinache/exorim_closure_phases.txt", cp, header=f"sep={sep} mas, PA={PA} deg, contrast={contrast}:1, pixels={pixels}, "
                                                                  f"plate_scale={plate_scale:5f} mas")
    plt.show()


if __name__ == "__main__":
    main()
