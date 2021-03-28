from ExoRIM.operators import Baselines
import numpy as np
import ExoRIM as exo
from ExoRIM.operators import closure_phase_covariance_inverse
from ExoRIM.definitions import rad2mas, mas2rad, triangle_pulse_f
from ExoRIM.log_likelihood import *
import matplotlib.pyplot as plt
import scipy.stats as st
import os


def fft_filter_hf(image):
    image_hat = np.fft.fft2(image)
    max_f = np.unravel_index(np.argsort(np.abs(image_hat)**2)[-5:][::-1], image_hat.shape)
    image_hat[max_f] *=0.1
    return np.abs(np.fft.ifft2(image_hat))


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

# test closure phase gradient for different plate scale


def main():
    np.random.seed(42)
    os.chdir("..")
    basedir = os.getcwd()
    results = os.path.join(basedir, "results", "experiment3")
    if not os.path.isdir(results):
        os.mkdir(results)

    N = 21
    L = 6
    pixels = 64
    wavel = 0.5e-6

    mask = np.random.normal(0, L, (N, 2))
    x = (L + np.random.normal(0, 1, N)) * np.cos(2 * np.pi * np.arange(N) / N)
    y = (L + np.random.normal(0, 1, N)) * np.sin(2 * np.pi * np.arange(N) / N)
    circle_mask = np.array([x, y]).T
    basel = Baselines(mask)
    b = np.sqrt(basel.UVC[:, 0]**2 + basel.UVC[:, 1]**2)
    sampled_scales = rad2mas(wavel/2/b)
    print(sampled_scales)
    bx = basel.UVC[:, 0]
    by = basel.UVC[:, 1]
    mL = np.max(np.sqrt(basel.UVC[:, 0]**2 + basel.UVC[:, 1]**2))
    minL = np.min(np.sqrt(basel.UVC[:, 0] ** 2 + basel.UVC[:, 1] ** 2))
    ideal_pl = rad2mas(1.22 * wavel / 4 / mL)
    theoretical_pl = rad2mas(1.22 * wavel/2/mL)
    print(f"Longest baseline {mL:.2f} m")
    print(f"Theoretical resolution {rad2mas(1.22* wavel/2/mL):.5f} mas")
    print(f"Ideal plate scale {rad2mas(wavel / 4 / mL):.5f} mas")
    plate_scale = np.linspace(rad2mas(1.22 * wavel / 4 / mL)*2/10, rad2mas(1.22 * wavel/4/mL)*2, 10)
    # plate_scale = np.array([0.6])
    m2pix = pixels * mas2rad(plate_scale) / wavel
    # print(m2pix)
    # print(np.sort(rad2mas(wavel / 2 / b)))
    # print(np.sort(rad2mas(wavel / 2 / bx)))
    # print(np.sort(rad2mas(wavel / 2 / by)))


    for pl in plate_scale:
        size = 5  # mas
        pixel_size = size / pl
        dx = -8 / pl
        dy = 8 / pl
        by = 3 / pl
        blur_kernel = gkern(5, 1)
        image_coords = np.arange(pixels) - pixels / 2.
        xx, yy = np.meshgrid(image_coords, image_coords)
        image1 = np.zeros_like(xx)
        rho = np.sqrt((xx) ** 2 + (yy) ** 2 / by)
        image1 += np.exp(-rho ** 2 / pixel_size ** 2)
        rho = np.sqrt((xx - dx) ** 2 + (yy + dy) ** 2 / by)
        image1 += np.exp(-rho ** 2 / pixel_size ** 2)
        # rho = np.sqrt((xx + 3) ** 2 + (yy + 5) ** 2)
        # image1 += np.exp(-rho**2 / 5 ** 2)
        # image1 = fftconvolve(image1, blur_kernel, "same")
        # rho = np.sqrt((xx) ** 2 + (yy) ** 2)
        # image1 += np.exp(-rho**2 / 5 ** 2)

        pixel_size = 4 / pl
        dx += 0
        dy += 2/pl
        noise = np.zeros_like(xx)
        noise += np.exp(-rho ** 2 / pixel_size ** 2)
        xx_prime = xx - dx
        yy_prime = yy + dy
        rho_prime = np.sqrt(xx_prime ** 2 + yy_prime ** 2 / by)
        noise += np.exp(-rho_prime ** 2 / pixel_size ** 2)

        SNR = 100
        phase_std = 1e-6

        B = exo.operators.Baselines(circle_mask)
        print(plate_scale)
        print(exo.definitions.rad2mas(1.22 * wavel / 4 / L))
        ndftm = exo.operators.NDFTM(B.UVC, wavel, pixels, pl)
        p = N * (N - 1) // 2
        q = (N - 1) * (N - 2) // 2
        # q = N * (N - 1) * (N - 2) // 6
        mycomplex = exo.definitions.MYCOMPLEX
        dtype = exo.definitions.DTYPE
        pulse = triangle_pulse_f(2 * np.pi * B.UVC[:, 0] / wavel, mas2rad(theoretical_pl)) * triangle_pulse_f(2 * np.pi * B.UVC[:, 1] / wavel, mas2rad(theoretical_pl))
        pulse = tf.constant(pulse.reshape((pulse.size, 1)), mycomplex)
        baselines = exo.operators.Baselines(mask_coordinates=mask)
        CPO = exo.operators.closure_phase_operator(baselines)
        # CPO = exo.operators.redundant_phase_closure_operator(baselines)
        # CPO = exo.operators.orthogonal_phase_closure_operator(baselines)
        # CPO = KPI(mask).uv_to_bsp
        sigma = tf.constant(closure_phase_covariance_inverse(CPO, phase_std), dtype)
        bisp_i = np.where(CPO != 0)
        V1_i = (bisp_i[0][0::3], bisp_i[1][0::3])
        V2_i = (bisp_i[0][1::3], bisp_i[1][1::3])
        V3_i = (bisp_i[0][2::3], bisp_i[1][2::3])
        V1_projector_np = np.zeros(shape=(q, p))
        V1_projector_np[V1_i] += 1.0
        V1_projector = tf.constant(V1_projector_np, dtype=mycomplex)
        V2_projector_np = np.zeros(shape=(q, p))
        V2_projector_np[V2_i] += 1.0
        V2_projector = tf.constant(V2_projector_np, dtype=mycomplex)
        V3_projector_np = np.zeros(shape=(q, p))
        V3_projector_np[V3_i] += 1.0
        V3_projector = tf.constant(V3_projector_np, dtype=mycomplex)
        CPO = tf.constant(CPO, dtype=dtype)
        A = tf.constant(ndftm, dtype=mycomplex) * pulse
        # Discrete Fourier Transform Matrices for bispectra
        A1 = tf.tensordot(V1_projector, A, axes=1)
        A2 = tf.tensordot(V2_projector, A, axes=1)
        A3 = tf.tensordot(V3_projector, A, axes=1)

        def bispectra(V):
            V1 = tf.einsum("ij, ...j -> ...i", V1_projector, V)
            V2 = tf.einsum("ij, ...j -> ...i", V2_projector, V)
            V3 = tf.einsum("ij, ...j -> ...i", V3_projector, V)
            return V1 * tf.math.conj(V2) * V3  # hack that works with baseline class! Be careful using other method

        V = tf.constant(ndftm.dot(image1.flatten()).reshape((1, -1)), mycomplex)
        Bisp = bispectra(V)
        phase_noise = np.random.normal(0, np.pi / 3, size=[B.nbap])
        imvec = tf.constant(noise.reshape((1, pixels, pixels, 1)))
        phstd = tf.constant(phase_std, dtype)
        snr = tf.constant(SNR, dtype)

        grad_cp = chisq_gradient_closure_phasor_analytic(imvec, A1, A2, A2, tf.math.angle(Bisp), phstd)
        grad_vis = chisq_gradient_complex_visibility_analytic(imvec, A, V, 1/snr)
        grad_vis_auto = chisq_gradient_complex_visibility_auto(imvec, A, V, 1/snr)
        grad_amp = chisq_gradient_amplitude_analytic(imvec, A, tf.math.abs(V), 1/snr)
        grad_bs = chisq_gradient_bispectra_analytic(imvec, A1, A2, A3, Bisp, 1/snr)
        grad_cp_v2 = chisq_gradient_closure_phase_auto(imvec, A, CPO, clphase=tf.math.angle(Bisp)%TWOPI, sigma=sigma)#phstd)
        grad_bs_auto = chisq_gradient_bispectra_auto(imvec, A1, A2, A3, Bisp, 1/snr)
        # grad_cp_hat = np.fft.fftshift(np.fft.fft2(grad_cp.numpy().reshape((pixels, pixels))))
        fig, axs = plt.subplots(3, 3, figsize=(20, 10), dpi=80)
        axs[0, 0].set_title("Ground Truth")
        axs[0, 0].imshow(image1, cmap="gray")
        axs[0, 1].set_title("Prediction")
        axs[0, 1].imshow(noise, cmap="gray")

        axs[0, 2].set_title(r"$\nabla_x \chi^2_{cp}$")
        im = axs[0, 2].imshow(noise*phase_std**2*grad_cp.numpy().reshape((pixels, pixels)))
        plt.colorbar(im, ax=axs[0, 2])

        axs[1, 0].set_title(r"$\nabla_x \chi^2_{vis}$")
        im = axs[1, 0].imshow(noise*1/SNR**2*grad_vis.numpy().reshape((pixels, pixels)))
        plt.colorbar(im, ax=axs[1, 0])

        axs[1, 1].set_title(r"$\nabla_x \chi^2_{visAuto}$")
        im = axs[1, 1].imshow(noise*1/SNR**2*grad_vis_auto.numpy().reshape((pixels, pixels)))
        plt.colorbar(im, ax=axs[1, 1])

        axs[1, 2].set_title(r"$\nabla_x \chi^2_{cpv2Auto}$")
        im = axs[1, 2].imshow(noise*phase_std**2*grad_cp_v2.numpy().reshape((pixels, pixels)))
        plt.colorbar(im, ax=axs[1, 2])

        axs[2, 0].set_title(r"$\nabla_x \chi^2_{bis}$")
        im = axs[2, 0].imshow(noise*phase_std**2*grad_bs.numpy().reshape((pixels, pixels)))
        plt.colorbar(im, ax=axs[2, 0])

        axs[2, 1].set_title(r"$\nabla_x \chi^2_{amp}$")
        im = axs[2, 1].imshow(noise*1/SNR**2*grad_amp.numpy().reshape((pixels, pixels)))
        plt.colorbar(im, ax=axs[2, 1])

        axs[2, 2].set_title(r"$\nabla_x \chi^2_{bisAuto}$")
        im = axs[2, 2].imshow(noise*phase_std**2*grad_bs_auto.numpy().reshape((pixels, pixels)))
        plt.colorbar(im, ax=axs[2, 2])

        plt.suptitle(f"plate_scale = {pl:.3f} mas, theoretical resolution = {theoretical_pl:.3f} mas")
        # plt.savefig(os.path.join(results, f"non_redundant_set_with_pulse_{pl:.3f}.png"))

        # fig, ax = plt.subplots()
        # ax.hist(np.abs(pulse))
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: f"{x/rad2mas(1):.2f}"))

        plt.show()
if __name__ == '__main__':
    main()