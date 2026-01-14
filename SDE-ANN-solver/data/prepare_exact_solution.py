# # coding=utf-8
# import tensorflow as tf
# import numpy as np
# from clasic_solver import euler_solver
#
# def load_paths(path="data/wieners.ckpt", dtype=tf.float64, n: int = 4):
#     N = 2 ** 12 if "12" in path else 2**17 # number of points per path
#     M = 10_000  # number of paths
#     paths = tf.Variable(tf.zeros(shape=(M, N), dtype=dtype), name="paths")
#     ckpt_restore = tf.train.Checkpoint(weights=paths)
#     ckpt_restore.restore(path).expect_partial()
#     paths = paths[:, ::int(N / n)]
#     return paths
#
# w = load_paths(path="wiener/wieners.ckpt", dtype=tf.float64, n=2 ** 17)
# print("loaded")
# time = tf.convert_to_tensor(np.linspace(0.0, 1., 2**17), dtype=tf.float64)
# result = tf.Variable(euler_solver(time, w), dtype=tf.float64)
# print("calculated")
# ckpt = tf.train.Checkpoint(weights=result)
# ckpt.write("euler/euler.ckpt")
# print(f"Saved euler paths to euler.ckpt with shape {result.shape} (float64)")


import tensorflow as tf
import numpy as np
from clasic_solver import euler_solver
import os

def load_paths_batch(path, dtype, n, start, end, total_M=10_000):
    """
    Loads a batch of Wiener paths [start:end] from checkpoint.
    This assumes that the checkpoint contains all M paths, but we slice only part of it.
    """
    N = 2 ** 12 if "12" in path else 2**17
    batch_size = end - start

    # Create a variable only for the batch slice
    paths = tf.Variable(tf.zeros(shape=(total_M, N), dtype=dtype), name="paths")
    ckpt_restore = tf.train.Checkpoint(weights=paths)
    ckpt_restore.restore(path).expect_partial()

    # Select only the batch slice and reduce its time resolution
    paths_batch = paths[start:end, ::int(N / n)]
    return paths_batch

def process_in_batches(path="wiener/wieners.ckpt",
                       dtype=tf.float64,
                       n=2**17,
                       batch_size=1000,
                       total_M=10_000,
                       save_path="euler/euler.ckpt", T: float = 1.0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Prepare time vector
    time = tf.convert_to_tensor(np.linspace(0.0, T, n), dtype=dtype)

    # Preallocate result checkpoint variable
    result = tf.Variable(tf.zeros(shape=(total_M, n), dtype=dtype), dtype=dtype)
    num_batches = (total_M + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total_M)
        print(f"Loading and processing batch {i + 1}/{num_batches} (paths {start}:{end})")

        # Load only the required subset of Wiener paths
        w_batch = load_paths_batch(path, dtype, n, start, end, total_M)
        # Process the batch
        result_batch = euler_solver(time, w_batch)
        # Save results into main result variable
        result[start:end].assign(result_batch)

        # Clean up
        del w_batch, result_batch
        tf.keras.backend.clear_session()

    # Save final checkpoint
    ckpt = tf.train.Checkpoint(weights=result)
    ckpt.write(save_path)
    print(f"✅ Saved Euler paths to {save_path} with shape {result.shape} (float64)")


def down_sample(path="euler/euler.ckpt",
                       dtype=tf.float64,
                       n=2**12,
                       batch_size=1000,
                       total_M=10_000,
                       save_path="euler_12/euler_12.ckpt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Preallocate result checkpoint variable
    result = tf.Variable(tf.zeros(shape=(total_M, n), dtype=dtype), dtype=dtype)
    num_batches = (total_M + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total_M)
        print(f"Loading and processing batch {i + 1}/{num_batches} (paths {start}:{end})")
        # Load only the required subset of Wiener paths
        w_batch = load_paths_batch(path, dtype, n, start, end, total_M)
        # Process the batch
        # Save results into main result variable
        result[start:end].assign(w_batch)
        # Clean up
        del w_batch
        tf.keras.backend.clear_session()
    # Save final checkpoint
    ckpt = tf.train.Checkpoint(weights=result)
    ckpt.write(save_path)
    print(f"✅ Saved Euler paths to {save_path} with shape {result.shape} (float64)")

if __name__ == "__main__":
    process_in_batches(batch_size=1000, T=1.0)
    down_sample(
        # path="wieners/wieners.ckpt",
        # save_path="wieners_12/wieners_12.ckpt",
        batch_size=1000)
    down_sample(
        path="wiener/wieners.ckpt",
        save_path="wiener_12/wieners_12.ckpt",
        batch_size=1000)