# coding=utf-8
import tensorflow as tf
import numpy as np

def get_batch_sampler(time: float, n_points: int, sigma: float, lam: float = 5., DL: bool = False, DW_AND_DL: bool = False):
    """Returns a function that generates input batches for a PINN (Physics-Informed Neural Network).

    Args:
        time: float
            Total time horizon T of the Brownian motion.
        n_points: int
            Number of discretization points along the trajectory.
        sigma: float
            Scaling factor for the Wiener process increments.

    Returns:
        sample_batch: callable
            A function `sample_batch(batch_size: int, random: bool = True)` that generates
            a batch of samples with shape [batch_size, n_points + 1].
    """
    MAX_JUNMPS = 20
    dt = time / (n_points - 1)
    def sample_batch(batch_size: int, random: bool = True, stack_with_time: bool = True):
        """
        Generates an input batch for the PINN.

        Args:
            batch_size: int
                Number of samples in the batch.
            random: bool, optional (default=True)
                If True, generates random time values t ∈ [0, T].
                If False, uses a uniform time grid over [0, T].

        Returns:
            x: tf.Tensor of shape [batch_size, n_points + 1]
                Tensor containing concatenated values [t, W₁, ..., Wₙ],
                where W represents a Wiener process (Brownian motion)
                sampled at `n_points` time steps.
        """
        # Generate Wiener trajectories
        if DL:
            Nj = np.random.poisson(lam * dt, size=(batch_size, n_points))
            jump_sizes = np.ones((batch_size, n_points, MAX_JUNMPS))
            mask = np.arange(MAX_JUNMPS) < Nj[..., None]
            dL = np.sum(jump_sizes * mask, axis=-1)
            dw = dL
        else:
            dw = np.random.randn(batch_size, n_points) * np.sqrt(dt)  # przyrosty W
        if DW_AND_DL:
            Nj = np.random.poisson(lam * dt, size=(batch_size, n_points))
            jump_sizes = np.random.uniform(
                low=-1,
                high=1,
                size=(batch_size, n_points, MAX_JUNMPS)
            )
            mask = np.arange(MAX_JUNMPS) < Nj[..., None]
            dL = np.sum(jump_sizes * mask, axis=-1)
            dw = dw + dL

        if sigma < 0.001:  # For sanity check only
            dw = sigma * dw
        w_samples = np.cumsum(dw, axis=1)
        if not stack_with_time:
            return tf.convert_to_tensor(w_samples, dtype=tf.float32)
        # Generate time samples
        if random:
            t = np.random.rand(batch_size, 1) * time  # losowy czas w [0,T]
        else:
            t = np.linspace(0, time, batch_size).reshape(-1, 1)  # siatka jednostajna

        # Combine time and Wiener process samples
        x = np.hstack([t, w_samples])  # [batch_size, N+1]
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x

    return sample_batch


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, None)),
        tf.TensorSpec(shape=(None, None)),
        tf.TensorSpec(shape=()),
    ]
)
def brownian_bridge_value(
        t: tf.Tensor, w_samples: tf.Tensor, end_time: float, random: bool = False
) -> tf.Tensor:
    """
    Computes the value (or expected value) of a Wiener process W(t)
    at time `t`, using a Brownian bridge between known sample points.

    Args:
        t: tf.Tensor of shape [batch_size, n]
            Time points at which to evaluate W(t).
        w_samples: tf.Tensor of shape [batch_size, n]
            Known values of the Wiener process at n time steps.
        end_time: float, Time boundary
        random: bool, optional (default=False)
            If True, returns random realizations sampled from the Brownian bridge.
            If False, returns the expected value of W(t).

    Returns:
        tf.Tensor of shape [batch_size, 1]
            - If `random=False`: the expected value of W(t).
            - If `random=True`: random samples from the Brownian bridge distribution.
    """
    n = tf.shape(w_samples)[1]
    time_grid = tf.linspace(0.0, end_time, n)

    idx = tf.searchsorted(time_grid, tf.squeeze(t), side='right') - 1
    idx = tf.clip_by_value(idx, 0, n - 2)

    t_i = tf.gather(time_grid, idx)
    t_ip1 = tf.gather(time_grid, idx + 1)

    w_i = tf.gather(w_samples, idx, batch_dims=1)
    w_ip1 = tf.gather(w_samples, idx + 1, batch_dims=1)

    # Linear interpolation
    alpha = (t - tf.expand_dims(t_i, -1)) / (tf.expand_dims(t_ip1 - t_i, -1))
    mean = tf.expand_dims(w_i, -1) + alpha * tf.expand_dims(w_ip1 - w_i, -1)
    if not random:
        return mean

    var = ((t - tf.expand_dims(t_i, -1)) *
           (tf.expand_dims(t_ip1, -1) - t) /
           (tf.expand_dims(t_ip1 - t_i, -1)))
    std = tf.sqrt(tf.maximum(var, 1e-12))
    w_t = mean + std * tf.random.normal(tf.shape(std))
    return w_t
