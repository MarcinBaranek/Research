# coding=utf-8
from dataclasses import dataclass
import tensorflow as tf
import keras


@dataclass(slots=True)
class TrainContext:
    optimizer: keras.optimizers.Optimizer
    loss_function: keras.losses.Loss


def get_train_step(context: TrainContext):
    @tf.function
    def train_step(model: keras.Model, x: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = context.loss_function(model, x)
        grads = tape.gradient(loss, model.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if
                      g is not None]
        context.optimizer.apply_gradients(grads_vars)
        return loss
    return train_step
