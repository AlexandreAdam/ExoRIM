import tensorflow as tf
from exorim import RIM, PhysicalModel, BaselineModel, Model, Modelv1, UnetModel


def test_baseline_model():
    model = BaselineModel()
    phys = PhysicalModel(pixels=32)
    rim = RIM(model, phys, time_steps=3)
    image = tf.random.normal(shape=(1, 32, 32, 1))
    X = phys.forward(image)
    rim(X)


def test_modelv1():
    model = Modelv1()
    phys = PhysicalModel(pixels=32)
    rim = RIM(model, phys, time_steps=3)
    image = tf.random.normal(shape=(1, 32, 32, 1))
    X = phys.forward(image)
    rim(X)


def test_modelv2():
    model = Model()
    phys = PhysicalModel(pixels=32)
    rim = RIM(model, phys, time_steps=3)
    image = tf.random.normal(shape=(1, 32, 32, 1))
    X = phys.forward(image)
    rim(X)


def test_unet_model():
    model = UnetModel()
    phys = PhysicalModel(pixels=32)
    rim = RIM(model, phys, time_steps=3)
    image = tf.random.normal(shape=(1, 32, 32, 1))
    X = phys.forward(image)
    rim(X)


if __name__ == '__main__':
    test_baseline_model()
