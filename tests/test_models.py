import tensorflow as tf
from ExoRIM.model import Model
import json


def test_call():
    with open("../hyperparameters_small.json", "r") as file:
        hparams = json.load(file)
    pix = hparams["pixels"]
    state_size = hparams["state_size"]
    state_depth = hparams["state_depth"]
    X = tf.random.normal(shape=(10, pix, pix, 1))
    h = tf.zeros(shape=(state_size, state_size, state_depth))
    model = Model(hparams)
    pred = model.call(X, h)


def test_save_and_load():
    with open("../hyperparameters_small.json", "r") as file:
        hparams = json.load(file)
    pix = hparams["pixels"]
    state_size = hparams["state_size"]
    state_depth = hparams["state_depth"]
    X = tf.random.normal(shape=(10, pix, pix, 1))
    h = tf.zeros(shape=(10, state_size, state_size, state_depth))
    model = Model(hparams)
    model(X, h)
    model.save_weights("model_test.h5")

    model = Model(hparams)
    model(X, h)
    model.load_weights("model_test.h5")

