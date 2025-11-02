# coding=utf-8
"""
Ornstein–Uhlenbeck (OU) process (SDE):
        dX_t = theta * (mu - X_t) dt + sigma dW_t

In words:
    - theta > 0 is the rate of mean reversion,
    - mu is the long-term mean,
    - sigma is the volatility coefficient,
    - W_t is a standard Wiener process (Brownian motion).
"""

import keras
import tensorflow as tf


sigma = 0.3 # > 0
theta = 1.1 # > 0
mu = -0.
T = 1.
N = 100

X_0 = -3.


@tf.function
def raw_a_function(x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    return theta * (mu - x)


@tf.function
def a_function(y: tf.Tensor, w: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    x = y + sigma * w
    return raw_a_function(x, t)



dX_0 = raw_a_function(X_0, 0)
