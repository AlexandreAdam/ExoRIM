import numpy as np
import tensorflow as tf
from ExoRIM.definitions import TWOPI
from ExoRIM.physical_model import MyopicPhysicalModel as PhysicalModel
from ExoRIM.log_likelihood import chi_squared_complex_visibility_with_self_calibration, chi_squared_complex_visibility, \
    cast_to_complex_flatten, chisq_gradient_complex_visibility_with_self_calibration_analytic,\
    chisq_gradient_complex_visibility_with_self_calibration_auto,\
    chisq_gradient_complex_visibility_analytic, chisq_gradient_complex_visibility_auto
from ExoRIM.rim import RIM
import matplotlib.pyplot as plt
import time
import os
import json

dtype = tf.float32
mycomplex = tf.complex64
results_dir = "../results/wizard/"
model_id = "20-08-27_11-55-46_RIM_vis"
weigths_file = os.path.join("../models", model_id, "rim_034_7.55257.h5")
data = os.path.join("../data/", model_id, "train", "image_0000_0020.txt")
h_file = os.path.join("../models", model_id, "hyperparameters_vis.json")
with open(h_file, "r") as f:
    hyperparameters = json.load(f)


def minimize_alpha( noise_tf, X,  phys, phi, phi_prime):
    step = 0
    lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1., decay_steps=300, power=3,
                                                       end_learning_rate=1e-5)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = chi_squared_complex_visibility_with_self_calibration(noise_tf, X, phys)
    previous_loss1 = 2*loss
    previous_loss2 = 2*loss
    delta = 100  # a percentage
    start = time.time()
    while step < 500 and delta > 1e-6:
        with tf.GradientTape() as tape:
            loss = chi_squared_complex_visibility_with_self_calibration(noise_tf, X, phys)
            delta = tf.math.abs(previous_loss2 - loss)/previous_loss2 * 100
            previous_loss2 = previous_loss1
            previous_loss1 = loss
        grads = tape.gradient(target=loss, sources=vars)
        grads = [tf.clip_by_norm(grads[0], 1)]
        opt.apply_gradients(zip(grads, vars))
        step += 1
        print(f"step={step} | loss = {loss:.4e} | lr = {opt.lr(step).numpy():.2e} ")
              # f"| diff ={np.mean(np.abs(phi - phi_prime - tf.einsum('ij, ...j -> ...i', phys.Bbar, vars[0]))) :.2e}")
        # if step % 20 == 0:
        #     print(1 - tf.math.cos(phi_prime - phi - tf.einsum('ij, ...j -> ...i', phys.Bbar, vars[0])).numpy())
    print(tf.reduce_mean(1 - tf.math.cos(phi_prime - phi - tf.einsum('ij, ...j -> ...i', phys.Bbar, vars[0])).numpy()))
    print(f"Took {time.time() - start:.2f} sec")
    return vars[0]


def main():
    N = 50
    pixels = 64
    mask = np.random.normal(0, 6, (N, 2))
    phys1 = PhysicalModel(pixels, mask, "visibility_with_self_cal")
    phys2 = PhysicalModel(pixels, mask, "visibility_with_self_cal", auto=True)
    rim1 = RIM(phys1, hyperparameters, weight_file=weigths_file)
    rim2 = RIM(phys2, hyperparameters, weight_file=weigths_file)

    x = np.arange(pixels) - pixels/2
    xx, yy = np.meshgrid(x, x)

    def rho(x0, y0):
        return np.sqrt((xx - x0)**2 + (yy - y0)**2)
    image = np.zeros_like(xx)
    image += np.exp(-rho(0, 0)**2 / 5**2)
    image += np.exp(-rho(10, 10)**2 / 7**2)
    image /= image.sum()
    image_tf = tf.constant(image, dtype, shape=(1, pixels, pixels, 1))
    pred1 = rim1.call(phys1.forward(image_tf))[0, ..., 0, -1].numpy()
    pred2 = rim2.call(phys2.forward(image_tf))[0, ..., 0, -1].numpy()
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Ground Truth")
    axs[1].imshow(pred1, cmap="gray")
    axs[1].set_title("Pred analytic")
    axs[2].set_title(pred2, cmap="gray")
    axs[2].set_title("Pred auto")

    plt.show()


    # losses = []
    # ideal_loss = []
    # step = 0
    # x_list = np.linspace(-20, 20, 20)
    # fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    # plt.ion()
    # alpha = tf.Variable(tf.random.normal(shape=(1, N - 1)), constraint=lambda x: x % TWOPI)
    # X = [0, 0, alpha]
    # for x in x_list:
    #     noise = np.zeros_like(xx)
    #     noise += np.exp(-rho(x, 0)**2/ 5**2)
    #     noise += np.exp(-rho(x + 10, x + 10)**2 / 7**2)
    #     noise /= noise.sum()
    #
    #     noise_tf = tf.constant(noise, dtype, shape=(1, pixels, pixels, 1))
    #
    #     amp = tf.math.abs(phys1.forward(image_tf))
    #     psi = tf.math.angle(phys1.bispectrum(image_tf))
    #
    #     phi = tf.math.angle(phys1.forward(noise_tf))
    #     phi_prime = tf.einsum("ij, ...j -> ...i", phys1.CPO_right_pseudo_inverse, psi)
    #     X = [amp, psi, X[-1]]
    #     # alpha = minimize(alpha, noise_tf, amp, psi, phys, phi, phi_prime)
    #     losses.append(chi_squared_complex_visibility_with_self_calibration(noise_tf, X, phys1).numpy())
    #     ideal_loss.append(chi_squared_complex_visibility(noise_tf, phys1.forward(image_tf), phys1).numpy())
    #     print(step)
    #     step += 1
    #
    #     plt.pause(1.e-6)
    #     axs[0, 0].clear()
    #     axs[0, 1].clear()
    #     axs[1, 0].clear()
    #     axs[1, 1].clear()
    #     noise = noise_tf.numpy()[0, ..., 0]
    #     axs[0, 0].imshow(image, cmap="gray")
    #     axs[0, 0].set_title("Ground Truth")
    #     axs[0, 1].imshow(noise, cmap="gray")
    #     axs[0, 1].set_title("Prediction")
    #     grad_analytic = chisq_gradient_complex_visibility_with_self_calibration_analytic(noise_tf, X, phys1).numpy()[0, ..., 0]
    #     axs[1, 0].imshow(grad_analytic)
    #     axs[1, 0].set_title("Self cal analytic")
    #     grad_auto = chisq_gradient_complex_visibility_with_self_calibration_auto(noise_tf, X, phys1).numpy()[0, ..., 0]
    #     axs[1, 1].imshow(grad_auto)
    #     axs[1, 1].set_title("Self cal auto")
    #     grad_ideal_analytic = chisq_gradient_complex_visibility_analytic(noise_tf, phys1.forward(image_tf), phys1)[0, ..., 0]
    #     grad_ideal_auto = chisq_gradient_complex_visibility_auto(noise_tf, phys1.forward(image_tf), phys1)[0, ..., 0]
    #     axs[2, 0].imshow(grad_ideal_analytic)
    #     axs[2, 0].set_title("Ideal analytic")
    #     axs[2, 1].imshow(grad_ideal_auto)
    #     axs[2, 1].set_title("Ideal auto")
    # print(ideal_loss)
    # losses = np.array(losses)
    # ideal_loss = np.array(ideal_loss)
    # plt.figure()
    # plt.plot(np.array(x_list), losses, "r.", label="Calibrated loss")
    # plt.plot(x_list, ideal_loss, "g.", label="Ideal loss")
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("Loss")
    # plt.yscale("log")
    # plt.savefig(os.path.join(results_dir, "calibrated_loss.png"))


    # alpha = tf.Variable(tf.random.normal(shape=(1, N - 1)), trainable=True, constraint=lambda alpha: alpha % TWOPI)
    # loss = lambda: chi_squared_complex_visibility_with_self_calibration(noise_tf, amp, psi, alpha, phys)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # print(f"L0 = {chi_squared_complex_visibility_with_self_calibration(noise_tf, amp, psi, alpha, phys).numpy():.2e}")
    # print(alpha.numpy())
    # opt = optimizer.minimize(loss, var_list=[alpha])
    # print(f"Lopt = {chi_squared_complex_visibility_with_self_calibration(noise_tf, amp, psi, alpha, phys).numpy():.2e}")
    # print(alpha.numpy())
    plt.pause(60)
    plt.show()


if __name__ == '__main__':
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    main()
