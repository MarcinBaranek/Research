# coding=utf-8
from os import write
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time

N = 2 ** 17   # number of points per path
M = 10_000    # number of paths
# N = 1000
# M = 10
T = 1.0       # time horizon

# def save_wiener_paths(path="wieners.ckpt", dtype=tf.float64, seed=42):
#     tf.random.set_seed(seed)
#     dt = T / N
#     stddev = tf.sqrt(tf.cast(dt, dtype))
#
#     dW = tf.random.normal(shape=(M, N), mean=0.0, stddev=stddev, dtype=dtype)
#
#     W = tf.cumsum(dW, axis=1)
#     W = tf.concat([tf.zeros((M, 1), dtype=dtype), W[:, :-1]], axis=1)
#     W = tf.Variable(W, dtype=dtype, name="W")
#     ckpt = tf.train.Checkpoint(weights=W)
#     ckpt.write(path)
#     print(f"Saved Wiener paths to {path} with shape {W.shape} ({dtype.name})")

LAM = 5.
max_jumps = 10

def save_wiener_paths_batch(path="wieners.ckpt", dtype=tf.float64, seed=42, batch_size=500):
    tf.random.set_seed(seed)
    dt = T / N
    stddev = tf.sqrt(tf.cast(dt, dtype))

    # allocate final tensor
    W_full = tf.Variable(tf.zeros((M, N), dtype=dtype), trainable=False)

    num_batches = (M + batch_size - 1) // batch_size

    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, M)
        current_batch = end - start

        # generate increments for the batch
        # dW = tf.random.normal(
        #     shape=(current_batch, N),
        #     mean=0.0,
        #     stddev=stddev,
        #     dtype=dtype
        # )

        Nj = tf.random.poisson(
            shape=(current_batch, N),
            lam=LAM * dt,
            # stddev=stddev,
            dtype=dtype
        )
        # jump_sizes = tf.random.uniform(
        #     shape=(current_batch, N, max_jumps),
        #     minval=-1,
        #     maxval=1,
        #     dtype=dtype
        # )
        jump_sizes = tf.constant(value=1.,
            shape=(current_batch, N, max_jumps),
            dtype=dtype
        )
        mask = tf.sequence_mask(
            Nj,
            maxlen=max_jumps,
            dtype=dtype
        )

        dL = tf.reduce_sum(jump_sizes * mask, axis=-1)

        # dW = dW + dL
        dW = dL

        # cumulative sum to build Wiener paths
        W_batch = tf.cumsum(dW, axis=1)

        # shift so W(0)=0
        W_batch = tf.concat(
            [tf.zeros((current_batch, 1), dtype=dtype), W_batch[:, :-1]],
            axis=1
        )

        # store into full tensor
        W_full[start:end].assign(W_batch)

        print(f"Processed batch {b+1}/{num_batches} (paths {start}–{end-1})")

    # print(f"{W_full.shape}")
    # plt.plot(W_full[0,:])
    # plt.plot(W_full[1,:])
    # plt.plot(W_full[2,:])
    # plt.show()
    # save checkpoint
    ckpt = tf.train.Checkpoint(weights=W_full)
    ckpt.write(path)
    print(f"\nSaved Wiener paths to {path} with shape {W_full.shape} ({dtype.name})")


if __name__ == "__main__":
    save_wiener_paths_batch()
