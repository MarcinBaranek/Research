# coding=utf-8
import keras
import tensorflow as tf
from sampler import brownian_bridge_value
from typing import Callable


def get_loss_function(
        time: float, a_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]
) -> Callable[[keras.Model, tf.Tensor], tf.Tensor]:
    """
    Creates and returns a loss function for training a Physics-Informed Neural Network (PINN)
    that models a stochastic differential equation using a Brownian bridge.

    Args:
        time: float
            Total time horizon (T) of the modeled process.
        a_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
            Function representing the right-hand side of the differential equation,
            typically depending on the model output and the Brownian bridge value.

    Returns:
        time_derivative_loss: Callable
            A function that computes the total loss for a given model and input tensor `x`.
    """
    def time_derivative_loss(model: keras.Model, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the PINN loss as the sum of:
            1. The time-derivative consistency term, and
            2. The difference between the learned time derivative and the
               right-hand side of the SDE defined via the Brownian bridge.

        Args:
            model: tf.keras.Model
                The neural network (e.g. a Physics-Informed Neural Network) used for prediction.
            x: tf.Tensor of shape [batch_size, n_points + 1]
                Input tensor containing concatenated values [t, W₁, …, Wₙ],
                where W represents Wiener process samples.

        Returns:
            tf.Tensor (scalar)
                The total loss value for the given batch.
        """
        # Part 1: Compute time derivative
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x, training=True)  # [batch_size, output_dim]
        dy_dx = tape.gradient(y, x)
        dy_dt = dy_dx[..., 0]  # [batch_size, output_dim]

        # Part 2: Compute Brownian bridge
        t = x[:, :1]
        w_samples = x[:, 1:]
        brownian_bridge = brownian_bridge_value(t, w_samples, tf.constant(time))

        # Part 3: PINN-style loss
        loss_term = tf.reduce_mean(tf.square(dy_dt - a_function(y, brownian_bridge, t)))
        total_loss = time * loss_term
        return total_loss
    return time_derivative_loss
