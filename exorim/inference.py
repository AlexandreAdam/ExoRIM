import tensorflow as tf
from .definitions import MYCOMPLEX, TWOPI, cast_to_complex_flatten


# ==========================================================================================
# (Reduced) Chi squared functions -- modified to assume diagonal covariance
# ==========================================================================================

def chi_squared_complex_visibility(image, X, phys, sigma):
    A = phys.A
    im = cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", A, im)
    diff = X - samples
    # if len(sigma.shape) < 2:
    chisq = 0.5 * tf.reduce_mean(tf.math.square(tf.math.abs(diff) / sigma), axis=1)
    # else:
    #     sigma = tf.cast(sigma, MYCOMPLEX)
    #     chisq = 0.5 * tf.einsum("...j, ...j -> ...", tf.einsum("...i, ij,  -> ...j", sigma, diff), tf.math.conj(diff))
    #     chisq /= X.shape[1]
    return chisq


def chi_squared_visibility_phases(image, X, phys, sigma):
    A = phys.A
    im = cast_to_complex_flatten(image)
    vphases_samples = tf.math.angle(tf.einsum("ij, ...j -> ...i", A, im)) % TWOPI
    diff = X - vphases_samples
    # if len(sigma.shape) < 2:
    chisq = tf.reduce_mean(tf.math.square(diff / sigma), axis=1)
    # else:
    #     chisq = tf.einsum("...i, ...i -> ...", tf.einsum("...i, ij -> ...j", diff, sigma), diff)
    #     chisq /= X.shape[1]
    return chisq


def chi_squared_amplitude(image, X, phys, sigma):
    A = phys.A
    im = cast_to_complex_flatten(image)
    amp_samples = tf.math.abs(tf.einsum("ij, ...j -> ...i", A, im))
    return tf.math.reduce_mean(((X - amp_samples)/sigma)**2, axis=1)


def chi_squared_amplitude_squared(image, X, phys, sigma):
    A = phys.A
    im = cast_to_complex_flatten(image)
    amp_samples = tf.math.abs(tf.einsum("ij, ...j -> ...i", A, im))
    return tf.math.reduce_mean(((X - amp_samples)/sigma)**2, axis=1)


def chi_squared_bispectra(image, X, phys, sigma):
    B_sample = phys.bispectrum(image)
    chisq = 0.5 * tf.reduce_mean(tf.math.square(tf.math.abs(X - B_sample)/sigma), axis=1)
    return chisq


def chi_squared_closure_phasor(image, X, phys, sigma):
    """
    The 2pi wrapping is taken into account by taking the likelihood
    of the absolute square difference of the closure phase phasors
    |e^(i*psi) - e^(i*psi')|^2.
    """
    clphase_samples = tf.math.angle(phys.bispectrum(image))
    chisq = tf.reduce_mean(((1 - tf.math.cos(X - clphase_samples)) / sigma)**2, axis=1)
    return chisq


def chi_squared_closure_phase(image, X, phys, sigma):
    """
    The phase wrapping in the range [0, 2pi) is enforced by the modulo operation.
    """
    im = cast_to_complex_flatten(image)
    phi = tf.math.angle(tf.einsum("ij, ...j -> ...i", phys.A, im)) % TWOPI
    clphase_sample = tf.einsum("ij, ...j -> ...i", phys.CPO, phi) % TWOPI
    diff = X - clphase_sample
    # if len(sigma.shape) < 2:
    chisq = 0.5 * tf.reduce_mean((diff/sigma)**2, axis=1)
    # else:
    #     chisq = 0.5 * tf.einsum("...i, ...i -> ...", tf.einsum("...i, ij -> ...j", diff, sigma), diff)
    return chisq


def chi_squared_append_amplitude_closure_phase(image, X, phys, sigma):
    chisq = chi_squared_amplitude(image, X[..., :phys.nbuv], phys, sigma[..., :phys.nbuv])
    chisq += chi_squared_closure_phasor(image, X[..., phys.nbuv:], phys, sigma[..., phys.nbuv:])
    return chisq


# ==========================================================================================
# Analytical Chi squared gradients
# ==========================================================================================


def chisq_gradient_complex_visibility(image, X, phys, sigma):
    sigma = tf.cast(sigma, MYCOMPLEX)
    im = cast_to_complex_flatten(image)
    samples = tf.einsum("ij, ...j -> ...i", phys.A, im)
    wdiff = (X - samples)/sigma**2
    out = -tf.math.real(tf.einsum("ji, ...j -> ...i", tf.math.conj(phys.A), wdiff))
    out = tf.reshape(out, image.shape)
    return out / X.shape[1]


def chisq_gradient_amplitude(image, X, phys, sigma):
    im = cast_to_complex_flatten(image)
    V_samples = tf.einsum("ij, ...j -> ...i", phys.A, im)
    amp_samples = tf.math.abs(V_samples)
    product = (X - amp_samples) / sigma**2 / amp_samples
    product = tf.cast(product, MYCOMPLEX)
    out = - 2.0 * tf.math.real(tf.einsum("ji, ...j -> ...i", tf.math.conj(phys.A), V_samples * product))
    out = tf.reshape(out, shape=image.shape)
    return out / X.shape[1]


def chisq_gradient_bispectra(image, X, phys, sigma):
    sigma = tf.cast(sigma, MYCOMPLEX)
    im = cast_to_complex_flatten(image)
    V1 = tf.einsum("ij, ...j -> ...i", phys.A1, im)
    V2 = tf.einsum("ij, ...j -> ...i", phys.A2, im)
    V3 = tf.einsum("ij, ...j -> ...i", phys.A3, im)
    B_samples = V1 * tf.math.conj(V2) * V3
    wdiff = tf.math.conj(X - B_samples) / sigma**2
    out = tf.einsum("ji, ...j -> ...i", phys.A1, wdiff * V2 * V3)
    out += tf.einsum("ji, ...j -> ...i", tf.math.conj(phys.A2), wdiff * tf.math.conj(V1 * V3))
    out += tf.einsum("ji, ...j -> ...i", phys.A3, wdiff * V1 * V2)
    out = -tf.math.real(out) / X.shape[1]
    out = tf.reshape(out, shape=image.shape)
    return out


def chisq_gradient_closure_phasor(image, X, phys, sigma):
    im = cast_to_complex_flatten(image)
    V1 = tf.einsum("ij, ...j -> ...i", phys.A1, im)
    V2 = tf.einsum("ij, ...j -> ...i", phys.A2, im)
    V3 = tf.einsum("ij, ...j -> ...i", phys.A3, im)
    B = V1 * tf.math.conj(V2) * V3
    clphase_samples = tf.math.angle(B)
    wdiff = tf.cast(tf.math.sin(X - clphase_samples) / sigma ** 2, MYCOMPLEX)
    out = tf.einsum("ji, ...j -> ...i", tf.math.conj(phys.A1), wdiff / tf.math.conj(V1))
    out = out + tf.einsum("ji, ...j -> ...i", phys.A2, wdiff / V2)
    out = out + tf.einsum("ji, ...j -> ...i", tf.math.conj(phys.A3), wdiff / tf.math.conj(V3))
    out = -2. * tf.math.imag(out) / B.shape[1]
    out = tf.reshape(out, shape=image.shape)
    return out


def chisq_gradient_append_amplitude_closure_phase(image, X, phys, sigma):
    grad = chisq_gradient_amplitude(image, X[..., :phys.nbuv], phys, sigma[..., :phys.nbuv])
    grad += chisq_gradient_closure_phasor(image, X[..., phys.nbuv:], phys, sigma[..., phys.nbuv:])
    return grad


# ==========================================================================================
# Complex Visibility Transformations
# ==========================================================================================

def v_to_bispectra(V, phys):
    V1 = tf.einsum("ij, ...j -> ...i", phys.V1, V)
    V2 = tf.einsum("ij, ...j -> ...i", phys.V2, V)
    V3 = tf.einsum("ij, ...j -> ...i", phys.V3, V)
    return V1 * tf.math.conj(V2) * V3


def v_to_closure_phasor(V, phys):
    V1 = tf.einsum("ij, ...j -> ...i", phys.V1, V)
    V2 = tf.einsum("ij, ...j -> ...i", phys.V2, V)
    V3 = tf.einsum("ij, ...j -> ...i", phys.V3, V)
    return tf.math.angle(V1 * tf.math.conj(V2) * V3)


def v_to_closure_phase(V, phys):
    return tf.einsum("ij, ...j -> ...i", phys.CPO, tf.math.angle(V) % TWOPI) % TWOPI


def append_amp_closure_phases(V, phys):
    amp = tf.math.abs(V)
    closure_phase = v_to_closure_phase(V, phys)
    return tf.concat([amp, closure_phase], axis=1)


# ==========================================================================================
# Computation graph
# ==========================================================================================

chi_squared = {
    "visibility": chi_squared_complex_visibility,
    "visibility_phase": chi_squared_visibility_phases,
    "visibility_amplitude": chi_squared_amplitude,
    "closure_phasor": chi_squared_closure_phasor,
    "closure_phase": chi_squared_closure_phase,
    "bispectra": chi_squared_bispectra,
    "append_visibility_amplitude_closure_phase": chi_squared_append_amplitude_closure_phase
}

chisq_gradients = {
    "visibility": chisq_gradient_complex_visibility,
    "visibility_amplitude": chisq_gradient_amplitude,
    "closure_phasor": chisq_gradient_closure_phasor,
    "bispectra": chisq_gradient_bispectra,
    "append_visibility_amplitude_closure_phase": chisq_gradient_append_amplitude_closure_phase
}

v_transformation = {
    "visibility": lambda V, phys: V,
    "visibility_phase": lambda V, phys: tf.math.angle(V),
    "visibility_amplitude": lambda V, phys: tf.math.abs(V),
    "closure_phasor": v_to_closure_phasor,
    "closure_phase": v_to_closure_phase,
    "bispectra": v_to_bispectra,
    "append_visibility_amplitude_closure_phase": append_amp_closure_phases,
    "append_visibility_real_imag": lambda V, phys: tf.concat([tf.math.real(V), tf.math.imag(V)], axis=1)
}
