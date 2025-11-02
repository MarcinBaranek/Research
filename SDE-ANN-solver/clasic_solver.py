import tensorflow as tf
from constants import raw_a_function, X_0, sigma


@tf.function(jit_compile=True)
def euler_solver(t: tf.Tensor, w_samples: tf.Tensor) -> tuple[tf.Tensor]:
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
        drift = raw_a_function(prev, t_i)
        return prev + drift * dt_i + sigma * dW_i

    # Stack inputs for scan
    t_seq = t[:-1]  # [n-1]
    dt_seq = dt  # [n-1]
    dW_seq = tf.transpose(dW, [1, 0])  # [n-1, batch_size]

    # Use tf.scan to propagate over time
    X = tf.scan(
        fn=lambda prev, inp: step(prev, inp),
        elems=(t_seq, dt_seq, dW_seq),
        initializer=tf.ones((batch_size,)) * tf.constant(X_0),
    )

    # Add initial condition at t=0
    X = tf.concat([tf.ones((1, batch_size)) * tf.constant(X_0), X], axis=0)
    return tf.transpose(X, [1, 0])  # [batch_size, n]