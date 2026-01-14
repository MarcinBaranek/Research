# coding=utf-8
import tensorflow as tf
import keras
from constants import sigma, T, X_0, dX_0, a_function
from evaluator import Evaluator
from model import Network
from loss import get_loss_function
from sampler import get_batch_sampler
from tqdm import tqdm
import matplotlib.pyplot as plt

tf.random.set_seed(7)
batch_size = 64
learning_rate = 1e-4
optimizer = keras.optimizers.Adam(learning_rate)

num_steps = 1_000

for N in [
    # 2**4, 2**5
    2**9,
    # 2**12,
    # 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13,
]:
    model = Network(N, X_0, float(dX_0), sigma, T)
    # model(tf.zeros([2, N + 1]))
    # model.load_weights(f"models/n{N}/model_{4000}.keras")
    loss_function = get_loss_function(T, a_function)
    sample_batch = get_batch_sampler(T, N, sigma)
    evaluator = Evaluator(N, T)


    @tf.function(reduce_retracing=True)
    def train_step(model: keras.Model, x: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = loss_function(model, x)
        grads = tape.gradient(loss, model.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if
                            g is not None]
        optimizer.apply_gradients(grads_vars)
        return loss

    # losses = open(f"results/n{N}/loss.txt", "a+")
    # l2_errors = open(f"results/n{N}/eval_l2.txt", "a+")
    # sup_errors = open(f"results/n{N}/eval_last.txt", "a+")
    #
    # save_each = 50
    model_save = 10000
    losses = []
    for step in tqdm(list(range(1, num_steps + 1))):
        x_batch = sample_batch(batch_size=batch_size, random=True)
        loss_val = train_step(model, x_batch)
        losses.append(loss_val)
        # if step % save_each == 0:
        #     result = evaluator.evaluate(model, max_eval_batch=64)
        #     losses.write(str(loss_val))
        #     l2_errors.write(str(result.l2_error))
        #     sup_errors.write(str(result.last_error))
        #     l2_errors.flush()
        #     sup_errors.flush()
        #     losses.flush()
        if step % model_save == 0:
            model.save(f"models/n{N}/model_{step}.keras")
    # plt.plot(losses[100:])
    plt.ylim(0, 100)
    plt.show()
    # with open(f"results/n{N}/loss.txt", "a+") as f:
    #     for l in losses:
    #         f.write(str(l))
