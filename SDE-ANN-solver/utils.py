# coding=utf-8
import keras
import tensorflow as tf
from logger import logger
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from constants import a_function, sigma, T, N
from clasic_solver import euler_solver


@dataclass(slots=True)
class ModelContext:
    model: keras.Model
    x_batch: tf.Tensor


def log_training_loss(step: int, loss_val: tf.Tensor, context: ModelContext):
    logger.info(f"Step {step:5d} | Loss: {loss_val.numpy():.6f}")
    with tf.GradientTape() as tape:
        tape.watch(context.x_batch)
        y = context.model(context.x_batch)
    dy_dx = tape.gradient(y, context.x_batch)
    grad_t_norm = tf.reduce_mean(
        tf.sqrt(tf.reduce_sum(dy_dx[..., 0] ** 2, axis=-1)))
    logger.info(f"∂y/∂t norm: {grad_t_norm.numpy():.5f}")


def plot_result(context: ModelContext, n_plots: int):
    n_plots = min(context.x_batch.shape[0], n_plots)
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 4 * n_plots))

    for i in range(n_plots):
        w = context.x_batch[i, 1:].numpy()
        t_grid = np.linspace(0, T, N).reshape(-1, 1)  # [N,1]
        x_input = np.hstack([t_grid, np.tile(w, (N, 1))])
        x_input_tf = tf.convert_to_tensor(x_input, dtype=tf.float32)  # [N, N+1]

        # calculate network putput and derivative
        with tf.GradientTape() as tape:
            tape.watch(x_input_tf)
            y = context.model(x_input_tf)  # [N,1]

        dy_dt = tape.gradient(y, x_input_tf)[..., 0]

        # calculate true derivative
        w_tensor = tf.convert_to_tensor(w.reshape(-1, 1),
                                        dtype=tf.float32)  # [N, 1]
        expected_drift = a_function(y, w_tensor, tf.convert_to_tensor(t_grid))  # [N,1]
        y = y + sigma * w_tensor

        # Plot derivatives
        axes[i, 0].plot(t_grid, dy_dt.numpy(),label="∂n/∂t (Network)")
        axes[i, 0].plot(
            t_grid, np.reshape(expected_drift.numpy(), newshape=(-1,1)),
            '--', label="Expected derivative (a_function)"
        )
        axes[i, 0].set_xlabel("time t")
        axes[i, 0].set_ylabel("Derivative")
        axes[i, 0].set_title("Network time derivative vs expected")
        axes[i, 0].legend()
        axes[i, 0].grid(True)

        # ===========================================================
        # Wykres trajektorii
        # ===========================================================
        X_true = euler_solver(
            tf.convert_to_tensor(t_grid, dtype=tf.float32),
            tf.convert_to_tensor(w.reshape(1, -1)),
        )
        X_true_numpy = np.reshape(X_true.numpy(), newshape=(-1, 1))
        axes[i, 1].plot(t_grid, X_true_numpy, label="Expected Process")
        axes[i, 1].plot(t_grid, y.numpy(), '--', label="Network (Prediction)")
        axes[i, 1].set_xlabel("Time t")
        axes[i, 1].set_ylabel("X(t)")
        axes[i, 1].set_title("Trajectory vs Prediction")
        axes[i, 1].legend()
        axes[i, 1].grid(True)

        plt.tight_layout()
    plt.show()