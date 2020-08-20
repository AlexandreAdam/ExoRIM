import tensorflow as tf
from ExoRIM.definitions import mycomplex, dtype, TWOPI

# ==========================================================================================
# Helper functions
# ==========================================================================================


def cast_to_complex_flatten(image):
    im = tf.dtypes.cast(image, mycomplex)
    im = tf.keras.layers.Flatten(data_format="channels_last")(im)
    return im


def bispectrum(image, A1, A2, A3):
    im = cast_to_complex_flatten(image)
    einsum = "ij, ...j -> ...i"
    V1 = tf.einsum(einsum, A1, im)
    V2 = tf.einsum(einsum, A2, im)
    V3 = tf.einsum(einsum, A3, im)
    B = V1 * tf.math.conj(V2) * V3
    return B

# ==========================================================================================
# Chi squared functions
# ==========================================================================================


def chi_squared_complex_visibility(image, A, vis, sigma):
    """
    Chi squared of the complex visibilities.
        image: batch of square matrix (3-tensor)
        A: 2-tensor of phasor for the Fourier Transform (NDFTM operator)
        vis: data product of visibilties (dtype must be mycomplex)
        sigma: (0,1,2)-tensor. Inverse of the covariance matrix of the visibilities.
    """
    im = cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", A, im)
    diff = vis - samples
    if len(sigma.shape) < 2:
        chisq = 0.5 * tf.reduce_mean(tf.math.square(tf.math.abs(diff) / sigma), axis=1)
    else:
        sigma = tf.cast(sigma, mycomplex)
        chisq = 0.5 * tf.einsum("...j, ...j -> ...", tf.einsum("...i, ij,  -> ...j", sigma, diff), tf.math.conj(diff))
        chisq /= vis.shape[1]
    return chisq


def chi_squared_visibility_phases(image, A, vphases, sigma):
    im = cast_to_complex_flatten(image)
    vphases_samples = tf.math.angle(tf.einsum("ij, ...j -> ...i", A, im)) % TWOPI
    diff = vphases - vphases_samples
    if len(sigma.shape) < 2:
        chisq = tf.reduce_mean(tf.math.square(diff / sigma), axis=1)
    else:
        chisq = tf.einsum("...i, ...i -> ...", tf.einsum("...i, ij -> ...j", diff, sigma), diff)
        chisq /= vphases.shape[1]
    return chisq


def chi_squared_amplitude(image, A, amp, sigma):
    sig = tf.cast(sigma, dtype)
    im = cast_to_complex_flatten(image)
    amp_samples = tf.math.abs(tf.einsum("ij, ...j -> ...i", A, im))
    return tf.math.reduce_mean(((amp - amp_samples)/sig)**2, axis=1)


def chi_squared_bispectra(image, A1, A2, A3, B, sigma):
    sig = tf.cast(sigma, dtype)
    B_sample = bispectrum(image, A1, A2, A3)
    chisq = 0.5 * tf.reduce_mean(tf.math.square(tf.math.abs(B - B_sample)/sig), axis=1)
    return chisq


def chi_squared_closure_phasor(image, A1, A2, A3, clphase, sigma):
    """
    negative log likelihood term for the closure phases, simplified for the diagonal covariance. The 2pi wrapping is
    taken into account by taking the likelihood of the absolute square difference of the closure phase phasors
    |e^(i*psi) - e^(i*psi')|^2.
        Ai are the Fourier Transform Matrix to the i baseline of the ijk closure triangle.
    """
    sig = tf.cast(sigma, dtype)
    clphase_samples = tf.math.angle(bispectrum(image, A1, A2, A3))
    chisq = tf.reduce_mean(((1 - tf.math.cos(clphase - clphase_samples)) / sig)**2, axis=1)
    return chisq


def chi_squared_closure_phase(image, A, CPO, clphase, sigma):
    """
    Closure phases chi squared with optional non-diagonal covariance matrix inverse (sigma). The chis squared depends
    directly on the closure phases, and not on the bispectrum phasors. Thus, the phase wrapping in the range [0, 2pi) is
    explicit in this derivation.
    Note that this version chooses to work with the linear operator instead of the Bispectrum.
        The A operator is the discrete fourier transform matrix.
        CPO is the closure phase operator.
    """
    im = cast_to_complex_flatten(image)
    phi = tf.math.angle(tf.einsum("ij, ...j -> ...i", A, im)) % TWOPI
    clphase_sample = tf.einsum("ij, ...j -> ...i", CPO, phi) % TWOPI
    diff = clphase - clphase_sample
    if len(sigma.shape) < 2:
        chisq = 0.5 * tf.reduce_mean((diff/sigma)**2, axis=1)
    else:
        chisq = 0.5 * tf.einsum("...i, ...i -> ...", tf.einsum("...i, ij -> ...j", diff, sigma), diff)
    return chisq

# ==========================================================================================
# Analytical Chi squared gradients
# ==========================================================================================


def chisq_gradient_complex_visibility_analytic(image, A, vis, sigma):
    """
    The analytical gradient of the Chi squared of the complex visibilities relative to the image pixels. This is the analytical
    version, which computes much faster than the AutoGrad version.

    This function only support diagonal covariance matrix, given in the form of a (0,1)-tensor. For a 2-tensor
    covariance matrix, AutoGrad should be used since this analytical version is no longer valid.
    """
    sig = tf.cast(sigma, mycomplex)  # prevent dividing by zero
    im = cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", A, im)
    wdiff = (vis - samples)/(sig**2)
    out = -tf.math.real(tf.einsum("ji, ...j -> ...i", tf.math.conj(A), wdiff))
    out = tf.reshape(out, image.shape)
    return out / vis.shape[1]


def chisq_gradient_amplitude_analytic(image, A, amp, sigma):
    """
    The analytical gradient of the complex visibility amplitude.

    This function only support diagonal covariance matrix, given in the form of a (0,1)-tensor. For a 2-tensor
    covariance matrix, AutoGrad should be used since this analytical version is no longer valid.
    """
    im = cast_to_complex_flatten(image)
    V_samples = tf.einsum("ij, ...j -> ...i", A, im)
    amp_samples = tf.math.abs(V_samples)
    product = (amp - amp_samples) / sigma**2 / amp_samples
    product = tf.cast(product, mycomplex)
    out = - 2.0 * tf.math.real(tf.einsum("ji, ...j -> ...i", tf.math.conj(A), V_samples * product))
    out = tf.reshape(out, shape=image.shape)
    return out / amp.shape[1]


def chisq_gradient_bispectra_analytic(image, A1, A2, A3, B, sigma):
    """
    The analytical gradient of the complex bispectra.

    This version only support diagonal covariance matrix, given in the form of a (0,1)-tensor. For a 2-tensor
    covariance matrix, AutoGrad should be used since this analytical version is no longer valid.
    """
    sig = tf.cast(sigma, mycomplex)
    im = cast_to_complex_flatten(image)
    V1 = tf.einsum("ij, ...j -> ...i", A1, im)
    V2 = tf.einsum("ij, ...j -> ...i", A2, im)
    V3 = tf.einsum("ij, ...j -> ...i", A3, im)
    B_samples = V1 * tf.math.conj(V2) * V3
    wdiff = tf.math.conj(B - B_samples) / sig**2
    out = tf.einsum("ji, ...j -> ...i", A1, wdiff * V2 * V3)
    out += tf.einsum("ji, ...j -> ...i", tf.math.conj(A2), wdiff * tf.math.conj(V1 * V3))
    out += tf.einsum("ji, ...j -> ...i", A3, wdiff * V1 * V2)
    out = -tf.math.real(out) / B.shape[1]
    out = tf.reshape(out, shape=image.shape)
    return out


def chisq_gradient_closure_phasor_analytic(image, A1, A2, A3, clphase, sigma):
    """

    This version only support diagonal covariance matrix, given in the form of a (0,1)-tensor. For a 2-tensor
    covariance matrix, AutoGrad should be used since this analytical version is no longer valid.
    """
    im = cast_to_complex_flatten(image)
    V1 = tf.einsum("ij, ...j -> ...i", A1, im)
    V2 = tf.einsum("ij, ...j -> ...i", A2, im)
    V3 = tf.einsum("ij, ...j -> ...i", A3, im)
    B = V1 * tf.math.conj(V2) * V3
    clphase_samples = tf.math.angle(B)
    wdiff = tf.cast(tf.math.sin(clphase - clphase_samples) / sigma ** 2, mycomplex)
    out = tf.einsum("ji, ...j -> ...i", tf.math.conj(A1), wdiff / tf.math.conj(V1))
    out = out + tf.einsum("ji, ...j -> ...i", A2, wdiff / V2)
    out = out + tf.einsum("ji, ...j -> ...i", tf.math.conj(A3), wdiff / tf.math.conj(V3))
    out = -2. * tf.math.imag(out) / B.shape[1]
    out = tf.reshape(out, shape=image.shape)
    return out

# ==========================================================================================
# AutoGrad Chi squared gradients
# ==========================================================================================


def chisq_gradient_complex_visibility_auto(image, A, vis, sigma):
    """
    The gradient of the Chi squared of the complex visibilities relative to the image pixels. This is the
    AutoGrad version.
    # TODO investigate the gradient mirror flip bug (probably due to the A conjugate matrix not taken in AutoGrad)
    """
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chi_squared_complex_visibility(image, A, vis, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


def chisq_gradient_amplitude_auto(image, A, amp, sigma):
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chi_squared_amplitude(image, A, amp, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


def chisq_gradient_visibility_phases_auto(image, A, vis, sigma):
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chi_squared_visibility_phases(image, A, vis, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


def chisq_gradient_bispectra_auto(image, A1, A2, A3, B, sigma):
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chi_squared_bispectra(image, A1, A2, A3, B, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


def chisq_gradient_closure_phasor_auto(image, A1, A2, A3, clphase, sigma):
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chi_squared_closure_phasor(image, A1, A2, A3, clphase, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


def chisq_gradient_closure_phase_auto(image, A, CPO, clphase, sigma):
    """
    CPO: Closure Phase Operator
    """
    with tf.GradientTape() as tape:
        tape.watch(image)
        chisq = chi_squared_closure_phase(image, A, CPO, clphase, sigma)
    gradient = tape.gradient(target=chisq, sources=image)
    return gradient


chi_squared = {
    "visibility": chi_squared_complex_visibility,
    "visibility_phase": chi_squared_visibility_phases,
    "visibility_amplitude": chi_squared_amplitude,
    "closure_phasor": chi_squared_closure_phasor,
    "closure_phase": chi_squared_closure_phase,
    "bispectra": chi_squared_bispectra
}

chisq_gradients = {
    "analytical_visibility": chisq_gradient_complex_visibility_analytic,
    "auto_visibility": chisq_gradient_complex_visibility_auto,
    "visibility_phase": chisq_gradient_visibility_phases_auto,
    "analytical_visibility_amplitude": chisq_gradient_amplitude_analytic,
    "auto_visibility_ampltiude": chisq_gradient_amplitude_auto,
    "analytical_closure_phasor": chisq_gradient_closure_phasor_analytic,
    "auto_closure_phasor": chisq_gradient_closure_phasor_auto,
    "closure_phase": chisq_gradient_closure_phase_auto,
    "analytical_bispectra": chisq_gradient_bispectra_analytic,
    "auto_bispectra": chisq_gradient_bispectra_analytic
}