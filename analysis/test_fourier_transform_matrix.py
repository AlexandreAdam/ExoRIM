# Stated objective: Compare NDFTM with analytical solution of the Fourier Transform for known objectex
import numpy as np
from scipy.special import j1
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ExoRIM.operators import NDFTM, Baselines
from ExoRIM.definitions import mas2rad, rad2mas
from matplotlib.ticker import FuncFormatter
from scipy.special import gamma
from scipy.stats import binned_statistic
import os

results_dir = "../results/experiment5"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

def main():
    # Defines physical variables
    np.random.seed(42)
    N = 21
    L = 6
    radius = 20 # pixels
    lam_circle = 4*radius
    pixels = 64
    wavel = 0.5e-6
    plate_scale = 0.6  # 2.2 * 1000 / 2048 * 3 # James Webb plate scale, #exo.definitions.rad2mas(1.22 * wavel / 4 / L)/10 # mas / pixel at the diffraction limit
    var = 5

    # mask
    x = (L + np.random.normal(0, var, N)) * np.cos(2 * np.pi * np.arange(N) / N)
    y = (L + np.random.normal(0, var, N)) * np.sin(2 * np.pi * np.arange(N) / N)
    circle_mask = np.array([x, y]).T
    mask = np.random.normal(0, L, (N, 2))
    # selected_mask = circle_mask
    selected_mask = mask

    d_theta = mas2rad(plate_scale)
    B = Baselines(selected_mask)
    rho = np.sqrt(B.UVC[:, 0]**2 + B.UVC[:, 1]**2)/wavel
    sampled_scales = rad2mas(1/2/rho)/plate_scale
    print(f"Largest structure: {sampled_scales.max():.0f}")
    print(f"Smallest structure: {sampled_scales.min():.0f}")
    print(f"xi = {(rad2mas(1/2/rho.min())/pixels)}")
    sampling_frequency = 1/plate_scale

    signal_frequency = 1/(mas2rad(plate_scale*lam_circle))
    nyquist_frequency = 0.5*signal_frequency # half of pixel/RAD, plate_scale is the sampling rate
    # Estimated sampling rate
    estimated_rate = np.diff(np.sort(rho), 1).mean()
    rho_th = np.linspace(rho.min(), rho.max(), 1000)

    image_coords = np.arange(pixels) - pixels / 2.
    xx, yy = np.meshgrid(image_coords, image_coords)
    image1 = np.zeros_like(xx)
    rho_squared = (xx) ** 2 + (yy) ** 2
    a = radius * mas2rad(plate_scale)
    image1 += np.sqrt(rho_squared) < radius

    A = NDFTM(B.UVC, wavel, pixels, plate_scale) * d_theta**2
    V = A.dot(image1.ravel())
    x = 2*np.pi*a*rho_th
    V_th = j1(x)/x * 2*np.pi * a**2
    V_th_f = interp1d(rho_th, np.abs(V_th))

    def freq2scale(x):
        return rad2mas(1 /2/ x)#/plate_scale

    def scale2freq(x):
        return 1/2/mas2rad(x)

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(r"Comparison between analytic and discrete FT of circ($2\rho/a$)")
    frame1 = fig.add_axes((.1, .3, .8, .6))
    _frame1 = frame1.twiny()
    _frame1.set_xlabel(r"$\theta$ [mas]")
    _frame1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{freq2scale(x):.1f}"))
    frame1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/0.5/a**2/2/np.pi:.1f}"))
    plt.plot(rho, np.abs(V), "ko", label=r"$\mathcal{F} X$")
    plt.plot(rho_th, np.abs(V_th), "r-", label=r"$a J_1(2\pi a \rho) / \rho$")
    # frame1.axvline(signal_frequency, color="b")
    plt.axvline(signal_frequency, color="b")
    plt.annotate(f"Circle radius = a = {rad2mas(a):.2f} mas", xy=(0.6, 0.8), xycoords="axes fraction", color="k")
    plt.annotate(r"Sampling frequency = %.2f mas$^{-1}$" % (sampling_frequency), xy=(0.6, 0.65), xycoords="axes fraction", color="b")
    plt.annotate(r"Nyquist frequency = %.2f mas$^{-1}$" % (nyquist_frequency/rad2mas(1)), xy=(0.6, 0.6), xycoords="axes fraction", color="k")
    frame1.set_ylabel(r"$|\gamma| = \frac{|V|}{a^2\pi }$")
    frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
    # plt.annotate("")
    plt.xlim(rho.min(), rho.max())
    plt.legend()

    frame2 = fig.add_axes((.1, .1, .8, .2))
    frame2.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: f"{x/rad2mas(1):.2f}"))
    plt.plot(rho, np.abs(np.abs(V) - V_th_f(rho))/V_th_f(rho) * 100, 'ok')
    plt.yscale("log")
    plt.ylabel("Error %")
    plt.xlabel(r"$\rho$ (mas$^{-1}$)")
    # plt.xlabel(r"Baseline (m)")

    plt.xlim(rho.min(), rho.max())
    plt.savefig(os.path.join(results_dir, "test_fourier_transform_circle.png"), bbox_inches="tight")

    # it is the baselines that are distributed with this pdf, so we change scale
    def rho_pdf(rho):
        x = rho*wavel
        return x/L * np.exp(-x**2/L**2/4) / 2

    plt.figure()
    theta = sampled_scales/plate_scale
    plt.hist(np.sort(sampled_scales)[:-3]/plate_scale, bins=50, density=True)
    # print(f"mean theta = {freq2scale(2*gamma(3/2)/wavel)/plate_scale:.2f}")
    # print(f"var theta = {freq2scale((4 - 4*gamma(3/2)**2)/wavel)/plate_scale:.2f}")
    # print(f"skewness theta = {freq2scale(8*gamma(5/2)/wavel)/plate_scale:.2f}")
    # plt.plot(theta, rho_pdf(rho/2), "ko")
    plt.axvline(rad2mas(2*a)/plate_scale, color="r")
    plt.axvline(pixels, color="g")
    plt.axvline(1, color="g")
    plt.xlabel(r"$\Delta \theta$ sampled [pixels]")
    plt.annotate(f"Circle diameter = {rad2mas(2*a):.2f} mas", xy=(0.65, 0.9), xycoords="axes fraction", color="r")
    plt.annotate(f"Image size = {pixels} pixels", xy=(0.65, 0.5), xycoords="axes fraction", color="g")
    plt.annotate(f"Median = {np.median(theta):.2f} pixels", xy=(0.65, 0.8), xycoords="axes fraction", color="k")
    plt.annotate(f"Mean = {np.mean(theta):.2f} pixels", xy=(0.65, 0.7), xycoords="axes fraction", color="k")
    plt.annotate(f"std = {np.std(theta):.2f} pixels", xy=(0.65, 0.6), xycoords="axes fraction", color="k")
    plt.title("Baseline sampling of image dimensions in pixels")
    plt.savefig(os.path.join(results_dir, "image_pixels_sampling.png"))

    plt.show()


if __name__ == '__main__':
    main()
