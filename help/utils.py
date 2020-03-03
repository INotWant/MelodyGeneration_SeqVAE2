import numpy as np
import tensorflow as tf


def flatten_maybe_padded_sequences(maybe_padded_sequences, lengths=None):
    """Flattens the batch of sequences, removing padding (if applicable).

    Args:
      maybe_padded_sequences: A tensor of possibly padded sequences to flatten,
          sized `[N, M, ...]` where M = max(lengths).
      lengths: Optional length of each sequence, sized `[N]`. If None, assumes no
          padding.

    Returns:
       flatten_maybe_padded_sequences: The flattened sequence tensor, sized
           `[sum(lengths), ...]`.
    """

    def flatten_unpadded_sequences():
        # The sequences are equal length, so we should just flatten over the first
        # two dimensions.
        return tf.reshape(maybe_padded_sequences,
                          [-1] + maybe_padded_sequences.shape.as_list()[2:])

    if lengths is None:
        return flatten_unpadded_sequences()

    def flatten_padded_sequences():
        indices = tf.where(tf.sequence_mask(lengths))
        return tf.gather_nd(maybe_padded_sequences, indices)

    return tf.cond(
        tf.equal(tf.reduce_min(lengths), tf.shape(maybe_padded_sequences)[1]),
        flatten_unpadded_sequences,
        flatten_padded_sequences)


def data_confusion(batch_size, fake_data, true_data):
    """Complete the following functions:
        num = batch_size // 2
        dataset = concat([fake_data[:num], true_data[:num]], axis=0), labels
        dataset = shuffle(dataset)
        inputs, labels = split(dataset)

    :return: a tuple of tensor op, (data_inputs, data_labels)
    """
    num = batch_size // 2
    data_inputs = tf.concat([fake_data[:num], true_data[:num]], axis=0)

    fake_labels = [[0, 1] for _ in range(num)]
    true_labels = [[1, 0] for _ in range(num)]
    data_labels = tf.convert_to_tensor(np.concatenate((fake_labels, true_labels), 0))
    data_labels = tf.to_float(data_labels)
    data_labels = tf.tile(tf.expand_dims(data_labels, axis=1), [1, tf.shape(data_inputs)[1], 1])

    data_all = tf.concat([data_inputs, data_labels], axis=-1)
    # shuffle
    # https://github.com/tensorflow/tensorflow/issues/6269
    data_all = tf.gather(data_all, tf.random.shuffle(tf.range(tf.shape(data_all)[0])))

    inputs, labels = tf.split(data_all, [-1, 2], axis=2)
    labels = labels[:, :1, :]
    labels = tf.squeeze(labels, axis=1)

    return inputs, labels


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.

    see: https://github.com/LantaoYu/SeqGAN/blob/master/discriminator.py

    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(tf.layers.dense(
                input_, size,
                name='highway_lin_%d' % idx,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.05)))

            t = tf.sigmoid(tf.layers.dense(
                input_, size,
                name='highway_gate_%d' % idx,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.05)) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def slerp(p0, p1, t):
    """Spherical linear interpolation.
    https://blog.csdn.net/u012947821/article/details/17136443
    """
    omega = np.arccos(
        np.dot(np.squeeze(p0 / np.linalg.norm(p0)),
               np.squeeze(p1 / np.linalg.norm(p1))))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1
