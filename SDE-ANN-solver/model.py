# coding=utf-8
import keras
from sampler import brownian_bridge_value


class Network(keras.Model):
    def __init__(
            self, n_points: int, initial_value: float,
            derivative_initial_value: float, sigma: float,
            end_time: float = 1.,
            **kwargs
    ):
        super().__init__(**kwargs)
        hidden_layers = [
            keras.layers.Dense(
                min(4*n_points, 4*512), activation="relu"
            ),
            keras.layers.Dense(
                min(2*n_points, 2*512), activation="relu"
            ),
            # keras.layers.Dense(
            #     min(2*n_points, 2*512), activation="relu"
            # ),
            keras.layers.Dense(
                min(n_points, 512), activation="tanh"
            ),
        ]
        self.end_time = end_time
        self.sigma = sigma
        self.n_points = n_points
        self.initial_value = initial_value
        self.derivative_initial_value = derivative_initial_value
        self.hidden_layers = hidden_layers
        self.out = keras.layers.Dense(1)

    def call(self, x, training=False):
        t = x[..., :1]
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        f_theta = self.out(h)
        if not training:
            make_trajectory = \
                self.sigma * brownian_bridge_value(t, x, self.end_time, random=False)
        else:
            make_trajectory = 0
        return self.initial_value\
            + self.derivative_initial_value * t\
            + f_theta * t ** 2 / 2.\
            + make_trajectory

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_points": self.n_points,
            "initial_value": self.initial_value,
            "derivative_initial_value": self.derivative_initial_value,
            "sigma": self.sigma,
            "end_time": self.end_time,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
