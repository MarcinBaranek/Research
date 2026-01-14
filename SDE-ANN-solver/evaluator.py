import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
from model import Network
from constants import a_function, sigma

from loader import load_paths
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class EvaluatorResult:
    l2_error: float
    last_error: float
    l2_var: float = None
    last_var: float = None


class Evaluator:
    def __init__(self, n: int, T: float = 1.):
        self.T = T
        self.n = n
        if n <= 2**12:
            self.wiener = load_paths(
                "data/wiener_12/wieners_12.ckpt", dtype=tf.float64, n=n
            )
            self.exact = load_paths(
                "data/euler_12/euler_12.ckpt", dtype=tf.float64, n=n
            )
        else:
            self.wiener = load_paths(
                "data/wiener/wieners.ckpt", dtype=tf.float64, n=n
            )
            self.exact = load_paths(
                "data/euler/euler.ckpt", dtype=tf.float64, n=n
            )
        self.wiener = tf.cast(self.wiener, dtype=tf.float32)

    def plot(self, model: keras.Model, n_plots: int = 4, offset = 0, k=14, one_plot=False):
        plt.rcParams.update({'font.size': 14})
        if one_plot:
            time = tf.convert_to_tensor(np.linspace(0.0, self.T, self.n),
                                        dtype=tf.float32)
            time = tf.reshape(time, [-1, 1])
            W = tf.concat(
                [tf.expand_dims(self.wiener[offset, :], axis=0) for _ in
                 range(self.n)],
                axis=0
            )
            x_input_tf = tf.concat([time, W], axis=1)
            with tf.GradientTape() as tape:
                tape.watch(x_input_tf)
                y = model.call(x_input_tf, training=True)  # [N,1]

            dy_dt = tape.gradient(y, x_input_tf)[..., 0]
            w_tensor = tf.convert_to_tensor(W[0, :].numpy().reshape(-1, 1),
                                            dtype=tf.float32)  # [N,1]
            expected_drift = a_function(y, w_tensor, 0)
            y = y + sigma * w_tensor
            y = np.reshape(y.numpy(), (-1))
            l2_err = (y - self.exact[offset, :].numpy()) ** 2
            last = np.sqrt(l2_err[-1])
            l2_err = np.sqrt(l2_err.mean())

            plt.plot(time.numpy(), self.exact[offset, :].numpy(),
                            '--', label="Exact solution",
                            color="green")
            plt.plot(time.numpy(), y, '--', label="Prediction",
                            color="blue")
            plt.xlabel("Time t")
            plt.ylabel("Value")
            plt.title(
                r"$err_{2,4096}([0,1])$" + f"={l2_err:.4f}" + r", $err_{2,4096}(T)$=" + f"{last:.4f}")
            plt.legend()
            plt.grid(True)
            plt.suptitle(f"Model performance after 5000 epochs")
            plt.savefig(f"performance_5000_ex1.pdf", bbox_inches="tight")
            # plt.title("ModelP ")
            plt.show()
            return

        fig, axes = plt.subplots(n_plots, 2,
                                 figsize=(12, 6 * n_plots))
        r = offset
        for i in range(n_plots):
            time = tf.convert_to_tensor(np.linspace(0.0, self.T, self.n),
                                        dtype=tf.float32)
            time = tf.reshape(time, [-1, 1])
            W = tf.concat(
                [tf.expand_dims(self.wiener[r+i*k,:], axis=0) for _ in range(self.n)],
                axis=0
            )
            x_input_tf = tf.concat([time, W], axis=1)
            with tf.GradientTape() as tape:
                tape.watch(x_input_tf)
                y = model.call(x_input_tf, training=True)  # [N,1]

            dy_dt = tape.gradient(y, x_input_tf)[..., 0]
            w_tensor = tf.convert_to_tensor(W[0,:].numpy().reshape(-1, 1),
                                            dtype=tf.float32)  # [N,1]
            expected_drift = a_function(y, w_tensor, 0)
            y = y + sigma * w_tensor
            y = np.reshape(y.numpy(), (-1))
            l2_err = (y - self.exact[r+i*k,:].numpy()) ** 2
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
            axes[i, 1].plot(time.numpy(), self.exact[r+i*k,:].numpy(), '--', label="Exact solution",
                            color="green")
            axes[i, 1].plot(time.numpy(), y, '--', label="Prediction",
                            color="blue")
            axes[i, 1].set_xlabel("Time t")
            axes[i, 1].set_ylabel("Value")
            axes[i, 1].set_title(r"$err_{4096}^{\mathcal{L}^2}$" + f"={l2_err:.4f}" + r", $err_{4096}^{\mathcal{L}^\infty}$=" + f"{last:.4f}")
            axes[i, 1].legend()
            axes[i, 1].grid(True)
        fig.suptitle(f"Model performance after 5000 epochs")
        plt.savefig(f"performance_5000_ex2.pdf", bbox_inches="tight")
        # plt.title("ModelP ")
        plt.show()

    # def evaluate(self, model: keras.Model) -> EvaluatorResult:
    #     time = tf.convert_to_tensor(np.linspace(0.0, self.T, self.n),
    #                                 dtype=tf.float32)
    #     batch_size = self.wiener.shape[0]
    #     time = tf.reshape(time, [-1, 1])
    #     total = tf.zeros(shape=[batch_size, 1], dtype=tf.float64)
    #     last = tf.zeros(shape=[batch_size, 1], dtype=tf.float64)
    #     for i, t in enumerate(time.numpy()):
    #         predictions = model.predict_on_batch(
    #             tf.concat(
    #                 [tf.constant(t[0], shape=[batch_size, 1]), self.wiener],
    #                 axis=1
    #             )
    #         )
    #         last = tf.square(
    #             tf.cast(predictions, dtype=tf.float64)\
    #             - tf.expand_dims(self.exact[:, i], axis=-1)
    #         )
    #         total = total + last
    #     traj_errors = tf.sqrt(total / self.n)
    #     traj_m, traj_var = tf.reduce_mean(traj_errors), tf.math.reduce_variance(traj_errors)
    #     last_square = tf.square(last)
    #     last_m, last_var = tf.reduce_mean(last_square), tf.math.reduce_variance(last)
    #     return EvaluatorResult(
    #         l2_error=traj_errors[0],
    #         l2_var=traj_var,
    #         last_error=last_m,
    #         last_var=last_var
    #     )

    def evaluate(self, model: keras.Model,
                 max_eval_batch=64) -> EvaluatorResult:
        """
        Evaluate the network on all trajectories in self.wiener
        in a memory-efficient and GPU-friendly way.

        Args:
            model: keras.Model to evaluate
            max_eval_batch: maximum number of trajectories processed at once
        Returns:
            EvaluatorResult with L2 trajectory error and last time point error
        """
        batch_size = self.wiener.shape[0]
        N = self.wiener.shape[1]  # number of points in wiener paths
        n = self.n  # number of evaluation time points

        # 1. Create time points: shape (n, 1)
        time = tf.linspace(0.0, self.T, n)
        time = tf.reshape(time, [n, 1])

        # Initialize accumulators
        total_sq_error = tf.zeros([batch_size, n], dtype=tf.float64)

        # Process trajectories in smaller batches
        for start in range(0, batch_size, max_eval_batch):
            end = min(start + max_eval_batch, batch_size)
            current_batch_size = end - start

            # 2. Extract wiener batch: shape (current_batch_size, N)
            wiener_batch = self.wiener[start:end]

            # 3. Broadcast time: shape (current_batch_size, n, 1)
            time_expanded = tf.broadcast_to(time, [current_batch_size, n, 1])

            # 4. Broadcast wiener: shape (current_batch_size, n, N)
            wiener_expanded = tf.expand_dims(wiener_batch, axis=1)
            wiener_expanded = tf.broadcast_to(wiener_expanded,
                                              [current_batch_size, n, N])
            # 5. Concatenate: shape (current_batch_size, n, 1+N)
            inputs = tf.concat([time_expanded, wiener_expanded], axis=-1)

            # 6. Flatten for model: (current_batch_size * n, 1+N)
            inputs_flat = tf.reshape(inputs, [-1, N + 1])
            # 7. Model prediction
            predictions_flat = model(inputs_flat)
            # 8. Reshape back: (current_batch_size, n, 1)
            predictions = tf.reshape(predictions_flat,
                                     [current_batch_size, n, 1])

            # predictions = predictions + sigma * wiener_batch

            # 9. Compute squared error
            exact_batch = self.exact[
                          start:end]  # shape (current_batch_size, n)
            sq_error_batch = tf.square(
                tf.cast(predictions, tf.float64) - tf.expand_dims(exact_batch,
                                                                  axis=-1))
            total_sq_error = tf.tensor_scatter_nd_update(
                total_sq_error,
                indices=tf.reshape(tf.range(start, end), [-1, 1]),
                updates=tf.squeeze(sq_error_batch, axis=-1)
            )

        # 10. L2 trajectory errors
        traj_errors = tf.reduce_mean(total_sq_error, axis=1)  # (batch_size,)
        traj_m = tf.sqrt(tf.reduce_mean(traj_errors))

        # 11. Last time point error
        last_sq = total_sq_error[:, -1]  # shape (batch_size,)
        # last_sq = tf.sqrt(last_sq)
        last_m = tf.sqrt(tf.reduce_mean(last_sq))

        return EvaluatorResult(
            l2_error=traj_errors[0],
            last_error=last_m,
        )

if __name__ == "__main__":
    N = 512
    E = 4000
    T = 1.0
    # keras.utils.custom_object_scope.
    model = Network(N, -0.3, 3.5, 0.61, T)
    model(tf.zeros([0, N+1]))
    model.load_weights(f"models/n{N}/model_{E}.keras")
    evaluator = Evaluator(N, T=T)
    evaluator.plot(model, n_plots=5, offset=10, k=1, one_plot=False)