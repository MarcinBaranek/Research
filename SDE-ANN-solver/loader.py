# coding=utf-8
import tensorflow as tf


def load_paths(path="data/wieners.ckpt", dtype=tf.float64, n: int = 4):
    N = 2 ** 12 if "12" in path else 2**17 # number of points per path
    M = 10_000  # number of paths
    paths = tf.Variable(tf.zeros(shape=(M, N), dtype=dtype))
    ckpt_restore = tf.train.Checkpoint(weights=paths)
    ckpt_restore.restore(path).expect_partial()
    paths = paths[:, ::int(N / n)]
    return paths
