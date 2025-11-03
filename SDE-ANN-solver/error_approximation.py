# coding=utf-8
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import keras
from clasic_solver import euler_solver
from sampler import get_batch_sampler
from logger import logger


@dataclass(frozen=True, slots=True)
class Constants:
    X_0: float
    T: float
    N: int
    sigma: float


class ErrorApproximation:
    def __init__(self, model: keras.Model, const: Constants):
        self.model = model
        self.const = const

    def perform_error_approximation(self, dense_factor: int, batch_size: int):
        N = self.const.N * dense_factor
        sampler = get_batch_sampler(
            time=self.const.T, n_points=N, sigma=self.const.sigma
        )
        wiener_samples = sampler(batch_size=batch_size, random=False, stack_with_time=False)
        time = tf.convert_to_tensor(np.linspace(0.0, self.const.T, N), dtype=tf.float32)
        exact_sol = euler_solver(time, wiener_samples)[:,::dense_factor]
        time = time[::dense_factor]
        wiener_samples = wiener_samples[:,::dense_factor]
        total = tf.zeros(shape=[batch_size, 1])
        last = tf.zeros(shape=[batch_size, 1])
        for i, t in enumerate(time.numpy()):
            predictions = self.model.predict_on_batch(
                tf.concat(
                    [tf.constant(t, shape=[batch_size, 1]),wiener_samples],
                    axis=1
                )
            )
            last = tf.square(predictions - tf.expand_dims(exact_sol[:,i],axis=-1))
            total = total + last

        traj_errors = tf.sqrt(total/(N / dense_factor))
        traj_m, traj_var = tf.reduce_mean(traj_errors), tf.math.reduce_variance(traj_errors)

        last_square = tf.square(last)
        last_m, last_var = tf.reduce_mean(last_square), tf.math.reduce_variance(last)
        logger.info(f"L2 over trajectory: {traj_m}+-{traj_var}")
        logger.info(f"|estX_T-X_T|:       {last_m}+-{last_var}")
