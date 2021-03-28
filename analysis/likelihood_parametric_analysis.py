import numpy as np
import matplotlib.pyplot as plt
from ExoRIM.definitions import mas2rad
import ExoRIM as exo
import tensorflow as tf
tf.keras.backend.set_floatx('float32')


# Defines physical variables
N = 21
L = 6
pixels = 32
wavel = 0.5e-6
plate_scale = 3.2 #2.2 * 1000 / 2048 * 3 # James Webb plate scale, #exo.definitions.rad2mas(1.22 * wavel / 4 / L)/10 # mas / pixel at the diffraction limit
var = 1
ISZ = 32  # number of pixels for the image
cwavel = 0.5e-6 # Wavelength [m]
pscale = 3.2 # plate scale [mas/pixel]
m2pix = mas2rad(pscale) * ISZ / cwavel  # [1/m] units for (u, v) Fourier space

image_coords = np.arange(pixels) - pixels / 2.
xx, yy = np.meshgrid(image_coords, image_coords)
image1 = np.zeros_like(xx)
rho_squared = (xx) ** 2 + (yy) ** 2
image1 += rho_squared**(1/2) < 5
# image1 += (xx - 30) ** 2 + (yy - 30) ** 2 < 10
# plt.imshow(image1, cmap="gray")


x = (L + np.random.normal(0, var, N)) * np.cos(2 * np.pi * np.arange(N) / N)
y = (L + np.random.normal(0, var, N)) * np.sin(2 * np.pi * np.arange(N) / N)
circle_mask = np.array([x, y]).T
image_coords = np.arange(pixels) - pixels / 2.
xx, yy = np.meshgrid(image_coords, image_coords)


def bispectra(V):
    V1 = tf.einsum("ij, ...j -> ...i", V1_projector, V)
    V2 = tf.einsum("ij, ...j -> ...i", V2_projector, V)
    V3 = tf.einsum("ij, ...j -> ...i", V3_projector, V)
    return V1 * tf.math.conj(V2) * V3  # hack that works with baseline class! Be careful using other method

def zero_pad(image):
    return np.pad(image, pixels//2, 'constant', constant_values=0)


def zero_pad_over_batch(images):
    out = np.zeros((images.shape[0], 2*pixels, 2*pixels))
    out[:, pixels//2:-pixels//2, pixels//2:-pixels//2] = images
    return out

def crop_image(image):
    return image[..., pixels//2:-pixels//2, pixels//2:-pixels//2]


B = exo.operators.Baselines(circle_mask)
print(plate_scale)
print(exo.definitions.rad2mas(1.22 * wavel / 4 / L))
ndftm = exo.operators.NDFTM(B.UVC, wavel, pixels, plate_scale)
ndftm_i = exo.operators.NDFTM(B.UVC, wavel, pixels, plate_scale, inv=True)
p = N * (N - 1) // 2
# q = (N - 1) * (N - 2) // 2
q = N * (N - 1) * (N - 2) // 6
mycomplex = exo.definitions.MYCOMPLEX
dtype = exo.definitions.DTYPE
baselines = exo.operators.Baselines(mask_coordinates=circle_mask)
# CPO = exo.operators.phase_closure_operator(baselines)
CPO = exo.operators.redundant_phase_closure_operator(baselines)
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
A = tf.constant(ndftm, dtype=mycomplex)
# Discrete Fourier Transform Matrices for bispectra
A1 = tf.tensordot(V1_projector, A, axes=1)
A2 = tf.tensordot(V2_projector, A, axes=1)
A3 = tf.tensordot(V3_projector, A, axes=1)

V = ndftm.dot(zero_pad(image1).flatten())
phase_noise = np.random.normal(0, np.pi/3, size=[B.nbap])
visibility_phase_noise = np.einsum("ij, ...j -> ...i", B.BLM, phase_noise)

noisy_V = V * np.exp(1j * visibility_phase_noise)
Bisp = bispectra(V)
NoisyBisp = bispectra(noisy_V)


bas = np.sqrt(B.UVC[:, 0]**2 + B.UVC[:, 1]**2)
# plt.plot(bas, np.abs(V), "ko")



def cast_to_complex_flatten(image):
    im = tf.dtypes.cast(image, mycomplex)
    im = tf.keras.layers.Flatten(data_format="channels_last")(im)
    return im

def chisq_amp(image, A, amp, sigma):
    """Visibility Amplitudes (normalized) chi-squared"""
    sig = tf.cast(sigma, dtype)
    im = cast_to_complex_flatten(image)
    #amp_samples = tf.math.abs(tf.einsum("ij, ...j -> ...i", A, im))
    amp_samples = tf.einsum("ij, ...j -> ...i", A, im)
    amp_samples = amp_samples #* tf.math.exp(-1j * tf.math.angle(amp_samples[0]))
    amp_samples = tf.math.abs(amp_samples)
    return tf.math.reduce_mean(((amp - amp_samples)/sig)**2, axis=1)

def chisqgrad_amp(image, V, sigma, floor=1e-6):
    """The gradient of the amplitude chi-squared"""
    pix = pixels # since we zero pad
    amp = tf.cast(tf.math.abs(V), dtype)
    im = cast_to_complex_flatten(image)
    V_samples = tf.einsum("ij, ...j -> ...i", A, im)
    amp_samples = tf.math.abs(V_samples)
    den = tf.cast((sigma + floor)**2, dtype)
    product = (amp - amp_samples) / den / amp_samples
    product = tf.cast(product, mycomplex)
    adjoint = tf.transpose(tf.math.conj(A))
    out = -2.0 * tf.math.real(tf.einsum("ij, ...j -> ...i", adjoint, V_samples * product))
    out = tf.reshape(out, shape=[-1, pix, pix, 1])
    return out / amp.shape[1]


def chisq_bis(image, B, sigma):
    """

    :param image: Image tensor
    :param Amatrices:
    :param B: Bispectrum projection matrix to project from complex visibilities to bispectrum ()
    :param bis:
    :param sigma:
    :return:
    """
    sig = tf.cast(sigma, dtype)
    im = cast_to_complex_flatten(image)
    V1 = tf.einsum("ij, ...j -> ...i", A1, im)
    V2 = tf.einsum("ij, ...j -> ...i", A2, im)
    V3 = tf.einsum("ij, ...j -> ...i", A3, im)
    B_sample = V1 * tf.math.conj(V2) * V3
    chisq = 0.5 * tf.reduce_mean(tf.math.square(tf.math.abs(B - B_sample)/sig), axis=1)
    return chisq

def chisqgrad_bis(image, B, sigma, floor=1e-6):
    """The gradient of the bispectrum chi-squared"""
    pix = pixels
    sig = tf.cast(sigma, mycomplex)
    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    t_einsum = "ji, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    B_samples = V1 * tf.math.conj(V2) * V3
    wdiff = tf.math.conj(B - B_samples)/(sig + floor)**2
    out = tf.einsum(t_einsum, A1, wdiff * V2 * V3) + tf.einsum(t_einsum, A2, wdiff * V1 * V3) + tf.einsum(t_einsum, A3, wdiff * V1 * V2)
    out = -tf.math.real(out) / B.shape[1]
    out = tf.reshape(out, shape=[-1, pix, pix, 1])
    return out


def chisq_cphase(image, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    sig = tf.cast(sigma, dtype)
    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    clphase_samples = tf.math.angle(V1 * tf.math.conj(V2) * V3)
    chisq = tf.reduce_mean(((1. - tf.math.cos(clphase-clphase_samples)) / sig)**2, axis=1)
    return chisq

def gradchisq_cphase_auto(image, vis, sigma):
    im = tf.constant(image, dtype)
    with tf.GradientTape() as tape:
        tape.watch(im)
        chisq = chisq_cphase(im, tf.math.angle(bispectra(vis)), sigma)
    return tape.gradient(chisq, im)

def chisqgrad_cphase(image, clphase, sigma):
    """The gradient of the closure phase chi-squared"""
    pix = pixels
    floor = tf.constant(1e-8, dtype=mycomplex)
    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    t_einsum = "ji, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    B = V1 * tf.math.conj(V2) * V3
    clphase_samples = tf.math.angle(B)
    wdiff = tf.cast(tf.math.sin(clphase - clphase_samples)/sigma**2, mycomplex)
    out = tf.einsum(t_einsum, tf.math.conj(A1), wdiff / tf.math.conj(V1))
    out = out + tf.einsum(t_einsum, A2, wdiff / V2)
    out = out + tf.einsum(t_einsum, tf.math.conj(A3), wdiff / tf.math.conj(V3))
    out = -2. * tf.math.imag(out) / B.shape[1]
    out = tf.reshape(out, shape=[-1, pix, pix, 1])
    return out

def chisq_vis(image, A, vis, sigma):
    sig = tf.cast(sigma, dtype)
    im = cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", A, im)
    chisq = 0.5 * tf.reduce_mean((tf.math.abs(samples - vis) / sig)**2, axis=1)
    return chisq

def gradchisq_vis_auto(image, vis, sigma):
    im = tf.constant(image, dtype)
    with tf.GradientTape() as tape:
        tape.watch(im)
        chisq = chisq_vis(im, A, vis, sigma)
    return tape.gradient(chisq, im)

def chisq_real_image(image, A, vis, sigma_r, sigma_i=None):
    if sigma_i is None:
        sigma_i = sigma_r
    sig_i = tf.cast(sigma_i, tf.float32)
    sig_r = tf.cast(sigma_r, tf.float32)
    im = exo.definitions.cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", A, im)
#     chisq = 0
    chisq_r = 0.5 * tf.reduce_mean(((tf.math.real(samples) - tf.cast(tf.math.real(vis), dtype)) / sig_r)**2, axis=1)
    chisq_i = 0.5 * tf.reduce_mean(((tf.math.imag(samples) - tf.cast(tf.math.imag(vis), dtype)) / sig_i)**2, axis=1)
    return (chisq_r, chisq_i)

def gradchisq_real_imag(image, A, vis, sigma_r, sigma_i=None):
    if sigma_i is None:
        sigma_i = sigma_r
    sig_i = tf.cast(sigma_i, tf.float64)
    sig_r = tf.cast(sigma_r, tf.float64)
    ima = tf.constant(image, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(ima)
#         im = exo.definitions.cast_to_complex_flatten(ima)
        samples = tf.einsum("ij, ...j -> ...i", tf.math.real(A), ima)
        chisq_r = 0.5 * tf.reduce_mean(((samples - tf.cast(tf.math.real(vis), tf.float64)) / sig_r)**2, axis=1)
    grad_real = tape.gradient(chisq_r, ima)
    with tf.GradientTape() as tape:
        tape.watch(ima)
#         im = exo.definitions.cast_to_complex_flatten(ima)
        samples = tf.einsum("ij, ...j -> ...i", tf.math.imag(A), tf.cast(ima, tf.float64 ))
        chisq_i = 0.5 * tf.reduce_mean(((samples - tf.cast(tf.math.imag(vis), tf.float64)) / sig_i)**2, axis=1)
    grad_imag = tape.gradient(chisq_i, ima)
    return grad_real, grad_imag


batch = 100
# im_copies = zero_pad(image1).flatten().reshape((1, 4*image1.shape[0]**2))
im_copies = image1.flatten().reshape((1, image1.shape[0]**2))
im_copies = np.tile(im_copies, [batch, 1])
V_copies = V.reshape((1, V.shape[0]))
V_copies = np.tile(V_copies, [batch, 1])
V_copies = tf.cast(V_copies, exo.definitions.MYCOMPLEX)

angle = np.array([5 * np.pi/4 for i in range(batch//2)] + [np.pi/4 for i in range(batch//2)])
r = np.array(list(range(batch//2))[::-1] + list(range(batch//2))) * pixels/batch
x_prime = r * np.cos(angle)
y_prime = r * np.sin(angle)
noise = np.zeros((batch, pixels, pixels))
for i in range(batch):
    rho = np.sqrt((xx - x_prime[i]) ** 2 + (yy - y_prime[i]) ** 2)
    noise[i] += rho < 5
# noise = zero_pad_over_batch(noise)
# noise = noise.reshape(batch, 4*pixels**2)
noise = noise.reshape(batch, pixels**2)
log_likelihood = chisq_vis(noise, ndftm, V_copies, 0.1).numpy()
llr, lli = chisq_real_image(noise, ndftm, V_copies, 0.1)
chi_amp = chisq_amp(noise, ndftm, tf.math.abs(V_copies), 0.1)
chi_bis = chisq_bis(noise, bispectra(V_copies), 0.1)
chi_cp = chisq_cphase(noise, tf.math.angle(bispectra(V_copies)), 1e-4)
plt.figure(figsize=(8, 8))
plt.plot(r[batch//2:], log_likelihood[batch//2:], color="k", label=r"$\chi^2_{vis}$")
plt.plot(-r[:batch//2], log_likelihood[:batch//2], color="k")
plt.plot(r[batch//2:], lli.numpy()[batch//2:], color="r")
plt.plot(-r[:batch//2], lli.numpy()[:batch//2], color="r", label=r"$\chi^2_{imag}$")
plt.plot(r[batch//2:], llr.numpy()[batch//2:], color="b")
plt.plot(-r[:batch//2], llr.numpy()[:batch//2], color="b", label=r"$\chi^2_{real}$")
plt.plot(r[batch//2:], chi_amp.numpy()[batch//2:], color="g")
plt.plot(-r[:batch//2], chi_amp.numpy()[:batch//2], color="g", label=r"$\chi^2_{amp}$")
plt.plot(r[batch//2:], chi_bis.numpy()[batch//2:]/1e5, color="m")
plt.plot(-r[:batch//2], chi_bis.numpy()[:batch//2]/1e5, color="m", label=r"$\chi^2_{bis}$")
plt.plot(r[batch//2:], chi_cp.numpy()[batch//2:]/1e4, "--", color="k")
plt.plot(-r[:batch//2], chi_cp.numpy()[:batch//2]/1e4 , "--", color="k", label=r"$\chi^2_{\psi}$")
plt.xlabel(f"Position radius")
plt.ylabel("Log likelihood")
plt.title(rf"Noise in form of a circle is placed at different radius along $\theta=\pi/4$")
# plt.text(1, 0.5, rf"Noise amplitude: $\sigma$ = {sigma}", fontsize=14, transform=plt.gcf().transFigure)
plt.legend()
plt.show()
# plt.savefig("chisq_vs_circle_different_radius_zoomed.png", bbox_inches="tight")
# Test with zero padding

sigma = 0.1

V_tf = tf.cast(tf.reshape(V, [1, -1]), mycomplex)
grad_r, grad_i = gradchisq_real_imag(noise, ndftm, V_tf, sigma)
grad_vis = gradchisq_vis_auto(noise, V_tf, sigma)
grad_amp = chisqgrad_amp(noise, V_tf, sigma)
grad_bis = chisqgrad_bis(noise, bispectra(V_tf), sigma)
grad_cp = chisqgrad_cphase(noise, tf.math.angle(bispectra(V_tf)), 1e-4)
grad_cp_auto = gradchisq_cphase_auto(noise, V_tf, 1e-4)

# plt.imshow((1.*grad_amp.numpy()[..., 0] + 0.000002*grad_cp.numpy()[..., 0])[40])
