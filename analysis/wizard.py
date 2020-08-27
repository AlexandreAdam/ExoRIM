import numpy as np
import tensorflow as tf
from ExoRIM.definitions import TWOPI
from ExoRIM.physical_model import MyopicPhysicalModel as PhysicalModel
from ExoRIM.log_likelihood import chi_squared_complex_visibility_with_self_calibration, chi_squared_complex_visibility
import matplotlib.pyplot as plt
import time
import os

dtype = tf.float32
mycomplex = tf.complex64
results_dir = "../results/wizard/"


def plot_i(noise, phi_ker, phi_prime, phys):
    plt.ion()
    plt.pause(1.e-6)
    plt.clf()
    im = cast_to_complex_flatten(noise)
    vis_samples = tf.einsum("ij, ...j -> ...i", phys.A, im)
    vis_samples = vis_samples * tf.complex(tf.math.cos(phi_ker), tf.math.sin(phi_ker))
    amp_samples = tf.math.abs(vis_samples).numpy()
    plt.plot()


def minimize(alpha, noise_tf, amp, psi, phys, phi, phi_prime):
    step = 0
    lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=10., decay_steps=300, power=3,
                                                       end_learning_rate=1e-5)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    vars = [tf.Variable(alpha, constraint=lambda alpha: alpha % TWOPI)]
    loss = chi_squared_complex_visibility_with_self_calibration(noise_tf, amp, psi, vars[0], phys)
    previous_loss1 = 2*loss
    previous_loss2 = 2*loss
    delta = 100  # a percentage
    start = time.time()
    while step < 500 and delta > 1e-6:
        with tf.GradientTape() as tape:
            loss = chi_squared_complex_visibility_with_self_calibration(noise_tf, amp, psi, vars[0], phys)
            delta = tf.math.abs(previous_loss2 - loss)/loss * 100
            previous_loss2 = previous_loss1
            previous_loss1 = loss
        grads = tape.gradient(target=loss, sources=vars)
        grads = [tf.clip_by_norm(grads[0], 1)]
        # print(grads[0])
        opt.apply_gradients(zip(grads, vars))
        step += 1
        # print(f"step={step} | loss = {loss:.4e} | lr = {opt.lr(step).numpy():.2e} ")
              # f"| diff ={np.mean(np.abs(phi - phi_prime - tf.einsum('ij, ...j -> ...i', phys.Bbar, vars[0]))) :.2e}")
        # if step % 20 == 0:
        #     print(1 - tf.math.cos(phi_prime - phi - tf.einsum('ij, ...j -> ...i', phys.Bbar, vars[0])).numpy())
    print(tf.reduce_mean(1 - tf.math.cos(phi_prime - phi - tf.einsum('ij, ...j -> ...i', phys.Bbar, vars[0])).numpy()))
    print(f"Took {time.time() - start:.2f} sec")
    return vars[0]


def main():
    N = 21
    pixels = 64
    mask = np.random.normal(0, 6, (N, 2))
    phys = PhysicalModel(pixels, mask, "visibility")
    x = np.arange(pixels) - pixels/2
    xx, yy = np.meshgrid(x, x)

    def rho(x0, y0):
        return np.sqrt((xx - x0)**2 + (yy - y0)**2)
    image = np.zeros_like(xx)
    image += np.exp(-rho(0, 0)**2 / 5**2)
    image += np.exp(-rho(10, 10)**2 / 7**2)
    image /= image.sum()
    image_tf = tf.constant(image, dtype, shape=(1, pixels, pixels, 1))

    losses = []
    ideal_loss = []
    step = 0
    x_list = list(range(-30, 30))
    for x in x_list:
        noise = np.zeros_like(xx)
        noise += np.exp(-rho(x, 0)**2/ 5**2)
        noise += np.exp(-rho(x + 10, x + 10)**2 / 7**2)
        noise /= noise.sum()

        noise_tf = tf.constant(noise, dtype, shape=(1, pixels, pixels, 1))

        amp = tf.math.abs(phys.forward(image_tf))
        psi = tf.math.angle(phys.bispectrum(image_tf))
        alpha = tf.random.normal(shape=(1, N - 1))

        phi = tf.math.angle(phys.forward(noise_tf))
        phi_prime = tf.einsum("ij, ...j -> ...i", phys.CPO_right_pseudo_inverse, psi)

        alpha = minimize(alpha, noise_tf, amp, psi, phys, phi, phi_prime)
        losses.append(chi_squared_complex_visibility_with_self_calibration(noise_tf, amp, psi, alpha, phys).numpy())
        ideal_loss.append(chi_squared_complex_visibility(noise_tf, phys.forward(image_tf), phys).numpy())
        print(step)
        step += 1
    print(ideal_loss)
    losses = np.array(losses)
    ideal_loss = np.array(ideal_loss)
    plt.figure()
    plt.plot(np.array(x_list), losses, "r.", label="Calibrated loss")
    plt.plot(x_list, ideal_loss, "g.", label="Ideal loss")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()
    # alpha = tf.Variable(tf.random.normal(shape=(1, N - 1)), trainable=True, constraint=lambda alpha: alpha % TWOPI)
    # loss = lambda: chi_squared_complex_visibility_with_self_calibration(noise_tf, amp, psi, alpha, phys)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # print(f"L0 = {chi_squared_complex_visibility_with_self_calibration(noise_tf, amp, psi, alpha, phys).numpy():.2e}")
    # print(alpha.numpy())
    # opt = optimizer.minimize(loss, var_list=[alpha])
    # print(f"Lopt = {chi_squared_complex_visibility_with_self_calibration(noise_tf, amp, psi, alpha, phys).numpy():.2e}")
    # print(alpha.numpy())


if __name__ == '__main__':
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    main()