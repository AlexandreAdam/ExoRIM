import tensorflow as tf
from exorim import RIM, PhysicalModel, Model


def test_model_pipeline():
    model = Model(
        filters=32,
        filter_scaling=2,
        layers=3
    )
    phys = PhysicalModel(pixels=32)
    rim = RIM(model, phys, time_steps=3)
    image = tf.random.normal(shape=(1, 32, 32, 1))
    X = phys.forward(image)
    sigma = tf.ones_like(X) * 1e-2
    rim(X, sigma)


    model = Model(
        filters=32,
        filter_scaling=1,
        layers=2,
        block_conv_layers=3
    )
    phys = PhysicalModel(pixels=32)
    rim = RIM(model, phys, time_steps=3)
    image = tf.random.normal(shape=(1, 32, 32, 1))
    X = phys.forward(image)
    sigma = tf.ones_like(X) * 1e-2
    rim(X, sigma)

