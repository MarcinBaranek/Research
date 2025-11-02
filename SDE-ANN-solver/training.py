# coding=utf-8
import tensorflow as tf
import keras
from model import Network
from constants import sigma, T, N, X_0, dX_0, a_function
from loss import get_loss_function
from sampler import get_batch_sampler
from utils import log_training_loss, ModelContext, plot_result


batch_size = 64
learning_rate = 1e-4
optimizer = keras.optimizers.Adam(learning_rate)

num_steps = 5_000
log_every = 100
plot_every = 100


model = Network(N, X_0, dX_0)
loss_function = get_loss_function(T, a_function)
sample_batch = get_batch_sampler(T, N, sigma)

@tf.function
def train_step(model, x):
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

    if step % log_every == 0:
        log_training_loss(step, loss_val, context)
    if step % plot_every == 0:
        plot_result(context, n_plots=4)

print("✅ Trening zakończony!")
