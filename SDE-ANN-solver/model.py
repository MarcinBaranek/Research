# coding=utf-8
import keras

class Network(keras.Model):
    # X_0 + (theta * (mu - X_0)) * t + f_theta * t ** 2 / 2.
    def __init__(self, n_points: int, initial_value: float, derivative_initial_value: float):
        super().__init__()
        hidden_layers = [
            keras.layers.Dense(4*n_points, activation="tanh"),
            keras.layers.Dense(2*n_points, activation="tanh"),
            keras.layers.Dense(n_points, activation="tanh"),
        ]
        self.initial_value = initial_value
        self.derivative_initial_value = derivative_initial_value
        self.hidden_layers = hidden_layers
        self.out = keras.layers.Dense(1)

    def call(self, x):
        t = x[..., :1]
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        f_theta = self.out(h)
        return self.initial_value\
            + self.derivative_initial_value * t\
            + f_theta * t ** 2 / 2.
