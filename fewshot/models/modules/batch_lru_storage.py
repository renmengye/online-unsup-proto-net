from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.batch_storage import BatchStorage


class BatchLRUStorage(BatchStorage):
  """Least used storage with batch"""

  def __init__(self, bsize, size, shape, decay=1.0, dtype=tf.float32):
    """Initialize a storage.
    Args:
      bsize: Batch size
      size: Number of slots
      shape: Content being stored
    """
    super(BatchLRUStorage, self).__init__(bsize, size, shape, dtype=dtype)
    self._usage = BatchStorage(bsize, size, [])  # [B, M]
    self._age = BatchStorage(bsize, size, [])  # [B, M]
    self._decay = decay
    self._mode = "exp"  # exp or inv

  def allocate_new_key(self, mask=None, one_hot=False):
    """Allocate a new key based on least usage.
    Args:
      mask: [B] binary masking on batch dimension.
    """
    if self._mode == "exp":
      # -------Old---------
      idx = tf.argmin(self._usage.all, axis=1)  # [B]
    elif self._mode == "inv":
      # -------New---------
      # TODO: remove hard code, learning to decay
      idx = tf.argmin(
          self._usage.all * (1 + self._age.all)**-0.1, axis=1)  # [B]
    else:
      assert False

    # Clear up existing content.
    self.setitem(idx, tf.zeros(self._shape, dtype=self._dtype), mask=mask)
    self._usage.setitem(idx, 0.0, mask=mask)
    self._age.setitem(idx, 0.0, mask=mask)

    if one_hot:
      idx_dense = tf.one_hot(idx, self._size)  # [B, M]
      if mask is not None:
        idx_dense *= mask[:, None]
      return idx_dense
    else:
      return idx

  def write_dense(self, key, value, mask=None, in_gate=None, forget_gate=None):
    """Write to the memory and change the usage.
    Args:
      key: [B, M] normalized address over all slots.
      value: [B, D] content vector.
      mask: [B] bool.
    """
    eps = 1e-6
    if mask is not None:
      key = key * tf.cast(mask[:, None], self._dtype)
    tail = [1] * (len(self._storage.shape) - 2)
    full_shape = [self._bsize, self._size] + tail
    key_ = tf.reshape(key, full_shape)  # [B, M, 1]
    value_ = tf.expand_dims(value, 1)  # [B, 1, ...]

    if self._mode == "exp":
      # -------Old---------
      usage2_ = tf.reshape(self._usage.all, full_shape)  # [B, M, 1, ...]
    elif self._mode == "inv":
      # -------New---------
      # TODO: remove hard code, learning to decay
      usage_ = self._usage.all * (1 + self._age.all)**-0.1
      usage2_ = tf.reshape(usage_, full_shape)  # [B, M, 1, ...]
    else:
      assert False

    denom = tf.maximum(usage2_ + key_, eps)
    if in_gate is None:
      in_gate = key_ / denom
    if forget_gate is None:
      forget_gate = usage2_ / denom
    self._storage = self._storage * forget_gate + value_ * in_gate

    if self._mode == "exp":
      # -------Old---------
      self._usage._storage += key  # [B, M]

    elif self._mode == "inv":
      # -------New---------
      reset = tf.greater(key, 0.5)  # [B, M]
      self._age._storage = tf.where(reset, 0.0, self._age._storage)
      self._usage._storage = tf.where(reset, usage_ + key,
                                      self._usage.all + key)
    else:
      assert False

  def write_sparse(self, key, value, mask=None, in_gate=None,
                   forget_gate=None):
    """Write to the memory and change the usage.
    Args:
      key: [B] normalized address over all slots.
      value: [B, D] content vector.
      mask: [B] bool.
    """
    u = self._usage[key]
    u_ = u[:, None]
    m = self.getitem(key)
    if in_gate is None:
      in_gate = 1.0 / (u_ + 1.0)
    if forget_gate is None:
      forget_gate = u_ / (u_ + 1.0)
    self.setitem(key, m * forget_gate + value * in_gate, mask=mask)
    self._usage.setitem(key, u + 1.0, mask=mask)
    self._age.setitem(key, 0.0, mask=mask)  # Reset age parameter.

  def decay(self):
    if self._mode == "exp":
      self._usage._storage *= self._decay
    usage = self._usage._storage
    age = self._age._storage
    used = tf.greater(usage, 0.0)
    self._age._storage = tf.where(used, age + 1.0, age)

  def get_states(self):
    return self.all, self.usage.all, self.age.all

  def set_states(self, *states):
    self._storage = tf.identity(states[0])
    self._usage._storage = tf.identity(states[1])
    self._age._storage = tf.identity(states[2])

  def reset_states(self):
    self._storage = tf.zeros_like(self._storage)
    self._usage._storage = tf.zeros_like(self._usage._storage)
    self._age._storage = tf.zeros_like(self._age._storage)

  @staticmethod
  def from_states(*states, decay=1.0):
    assert len(states) == 3
    shape = states[0].shape
    s = BatchLRUStorage(
        shape[0],
        shape[1],
        list(shape[2:]),
        decay=decay,
        dtype=states[0].dtype)
    s._storage = tf.identity(states[0])
    s._usage._storage = tf.identity(states[1])
    s._age._storage = tf.identity(states[2])
    return s

  @property
  def usage(self):
    return self._usage

  @property
  def age(self):
    return self._age
