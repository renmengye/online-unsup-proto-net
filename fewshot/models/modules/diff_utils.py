import tensorflow as tf


def round_st(sigm):
  """Rount straight-through."""
  return tf.cast(tf.greater(sigm, 0.5),
                 sigm.dtype) - tf.stop_gradient(sigm) + sigm


def sample_gumbel(shape, eps=1e-20):
  U = tf.random.uniform(shape)
  return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax(y / temperature, axis=-1)


def gumbel_softmax(logits, temperature, hard=False):
  """
  ST-gumple-softmax
  input: [*, n_class]
  return: flatten --> [*, n_class] an one-hot vector
  """
  y = gumbel_softmax_sample(logits, temperature)
  if not hard:
    return y
  else:
    idx = tf.argmax(y, axis=-1)
    y_hard = tf.one_hot(idx, tf.shape(logits)[-1])
    y_hard = tf.stop_gradient(y_hard - y) + y
    return y_hard
