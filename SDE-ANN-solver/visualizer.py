# coding=utf-8
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


from constants import Constants


@dataclass
class Visualizer:
    constants: Constants

    def euler_solver(self, t: tf.Tensor, w_samples: tf.Tensor) -> tuple[tf.Tensor]:
        """
        Solve SDE using Euler method.

        Solves:
            dX_t = a(t, X_t) dt + sigma dW_t

        Args:
            t: tf.Tensor of shape [1, n]
                Common time grid for all trajectories.
            w_samples: tf.Tensor of shape [batch_size, n]
                Known values of the Wiener process at n coarse time steps.

        Returns:
            X: tf.Tensor of shape [batch_size, n]
                Simulated paths.
        """
        batch_size = tf.shape(w_samples)[0]
        t = tf.reshape(t, [-1])  # [n]
        dt = t[1:] - t[:-1]  # [n-1]
        dW = w_samples[:, 1:] - w_samples[:, :-1]  # [batch_size, n-1]
        # Function for one Euler step
        def step(prev, inputs):
            t_i, dt_i, dW_i = inputs
            drift = self.constants.a_function(prev, t_i)
            return prev + drift * dt_i + self.constants.sigma * dW_i

        # Stack inputs for scan
        t_seq = t[:-1]  # [n-1]
        dt_seq = dt  # [n-1]
        dW_seq = tf.transpose(dW, [1, 0])  # [n-1, batch_size]
        # Use tf.scan to propagate over time
        X = tf.scan(
            fn=lambda prev, inp: step(prev, inp),
            elems=(t_seq, dt_seq, dW_seq),
            initializer=tf.ones((batch_size,), dtype=self.constants.dtype) * self.constants.X_0,
        )
        # Add initial condition at t=0
        X = tf.concat([tf.ones((1, batch_size), dtype=self.constants.dtype) * tf.constant(self.constants.X_0), X], axis=0)
        result = tf.transpose(X, [1, 0])    # [batch_size, n]
        return result

    def visualize(self, wieners: tf.Tensor, model: keras.Model):
        time = tf.convert_to_tensor(
            np.linspace(0.0, self.constants.T, self.constants.N),
            dtype=wieners.dtype
        )
        time = tf.reshape(time, (1, -1))
        solutions = self.euler_solver(time, wieners)

        fig, axes = plt.subplots(wieners.shape[0], 2,
                                 figsize=(12, 6 * wieners.shape[0]))
        for i in range(wieners.shape[0]):
            time = tf.convert_to_tensor(np.linspace(0.0, self.constants.T, self.constants.N),
                                        dtype=tf.float32)
            time = tf.reshape(time, [-1, 1])
            W = tf.concat(
                [tf.expand_dims(wieners[i,:], axis=0) for _ in range(self.constants.N)],
                axis=0
            )
            x_input_tf = tf.concat([time, W], axis=1)
            with tf.GradientTape() as tape:
                tape.watch(x_input_tf)
                y = model.call(x_input_tf, training=True)  # [N,1]

            dy_dt = tape.gradient(y, x_input_tf)[..., 0]
            w_tensor = tf.convert_to_tensor(W[0,:].numpy().reshape(-1, 1),
                                            dtype=tf.float32)  # [N,1]
            expected_drift = self.constants.f_function(y, w_tensor, 0)
            y = y + self.constants.sigma * w_tensor
            y = np.reshape(y.numpy(), (-1))
            l2_err = (y - solutions[i,:].numpy()) ** 2
            last = np.sqrt(l2_err[-1])
            l2_err = np.sqrt(l2_err.mean())



            axes[i, 0].plot(time.numpy(), dy_dt.numpy(), label="∂Netowrk/∂t",
                            color="blue")
            axes[i, 0].plot(time.numpy(), expected_drift.numpy(), '--',
                            label="Expected derivative",
                            color="green")
            axes[i, 0].set_xlabel("Time t")
            axes[i, 0].set_ylabel("Value")
            axes[i, 0].set_title("Network derivative over time vs expected")
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            axes[i, 1].plot(time.numpy(), solutions[i,:].numpy(), '--', label="Exact solution",
                            color="green")
            axes[i, 1].plot(time.numpy(), y, '--', label="Prediction",
                            color="blue")
            axes[i, 1].set_xlabel("Time t")
            axes[i, 1].set_ylabel("Value")
            axes[i, 1].set_title(r"$err_{2,n}([0,T])$" + f"={l2_err:.4f}" + r", $err_{2,n}(T)$=" + f"{last:.4f}")
            axes[i, 1].legend()
            axes[i, 1].grid(True)
        plt.show()