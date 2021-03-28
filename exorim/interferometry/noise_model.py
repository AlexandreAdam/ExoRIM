import tensorflow as tf
import tensorflow_probability as tfp


class NoiseModel:
    def __init__(self, chi_term, SNR, form="polar"):
        self.chi_term = chi_term
        self.form = form
        assert form in ["polar", "euclidian"], f"form either <polar> or <euclidian>"

