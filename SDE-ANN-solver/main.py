# coding=utf-8
import keras
import tensorflow as tf

from logger import logger
from model import Network
from constants import sigma, T, X_0, dX_0, a_function
from loss import get_loss_function
from sampler import get_batch_sampler
from utils import ModelContext
from error_approximation import Constants, ErrorApproximation


for N in [
    # 4, 8, 16, 32, 64, 128, 256, 512, 2**10,
    #       2**11,
    # 2**12, 2**13, 2**14, 2**15, 2**16, 2**17
          ]:
    logger.info(f"N={N}")
    batch_size = 64
    learning_rate = 1e-4
    optimizer = keras.optimizers.Adam(learning_rate)

    num_steps = 5_000
    log_every = 10_00
    plot_every = 10_00

    model = Network(N, X_0, dX_0, sigma, T)
    loss_function = get_loss_function(T, a_function)
    sample_batch = get_batch_sampler(T, N, sigma)

    const = Constants(X_0, T, N, sigma)

    @tf.function(reduce_retracing=True)
    def train_step(model: keras.Model, x: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = loss_function(model, x)
        grads = tape.gradient(loss, model.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if
                        g is not None]
        optimizer.apply_gradients(grads_vars)
        return loss

    for step in range(1, num_steps + 1):
        x_batch = sample_batch(batch_size=batch_size, random=True)
        loss_val = train_step(model, x_batch)
        context = ModelContext(model, x_batch)
        # if step % log_every == 0:
        #     log_training_loss(step, loss_val, context)
        # if step % plot_every == 0:
        #     plot_result(context, n_plots=4)
    logger.info(f"Training finished {N}")
    error_calculator = ErrorApproximation(model, const)
    error_calculator.perform_error_approximation(10, 10_000)
