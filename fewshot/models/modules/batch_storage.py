"""Implements a FIFO storage.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule


class BatchStorage(ContainerModule):

  def __init__(self, bsize, size, shape, dtype=tf.float32):
    """Initialize a storage.
    Args:
      bsize: Batch size
      size: Number of slots
      shape: Content being stored
    """
    super(BatchStorage, self).__init__(dtype=dtype)
    if not isinstance(shape, list):
      shape = [shape]
    self._storage = tf.zeros([bsize, size] + shape, dtype=dtype)
    self._bsize = bsize
    self._size = size
    self._shape = shape
    self._dtype = dtype

  def __getitem__(self, key):
    """Gets an item.

    Args:
      key: [B] Indices.
    """
    return self.getitem(key)

  def getitem(self, key, mask=None):
    """Gets an item.

    Args:
      key: [B] Indices.
      mask: [B] Bool.
    """
    if isinstance(key, slice):
      assert False, "Not supported"
    elif isinstance(key, tuple):
      assert False, "Not supported"

    B = key.shape[0]
    bidx = tf.range(B, dtype=tf.int64)  # [B]
    key_ = tf.cast(tf.math.mod(key, self._size), tf.int64)
    tidx = tf.ones([B], dtype=bidx.dtype) * key_
    eidx = tf.stack([bidx, tidx], axis=1)  # [B, 2]
    if mask is not None:
      mask = tf.cast(mask, tf.bool)
      eidx = eidx[mask]  # [B', 2]
    return tf.gather_nd(self._storage, eidx)

  def __setitem__(self, key, value):
    """Sets the value in the storage.

    Args:
      key: [B]
      value: [B, ...]
    """
    self.setitem(key, value)

  def setitem(self, key, value, mask=None):
    """Sets the value in the storage.

    Args:
      key: [B]
      value: [B, ...]
      mask: [B] Bool.
    """
    B = key.shape[0]
    bidx = tf.range(B, dtype=tf.int64)  # [B]
    key_ = tf.cast(tf.math.mod(key, self._size), tf.int64)
    tidx = tf.ones([B], dtype=bidx.dtype) * key_
    eidx = tf.stack([bidx, tidx], axis=1)
    entry_old = tf.gather_nd(self._storage, eidx)
    entry_new = tf.scatter_nd(eidx, -entry_old + value, self._storage.shape)
    if mask is not None:
      mask = tf.cast(mask, self._dtype)
      new_shape = [-1] + [1] * (len(self._storage.shape) - 1)
      mask_ = tf.reshape(mask, new_shape)
      entry_new *= mask_
    self._storage = self._storage + entry_new

  def setitem_batch(self, key_batch, value_batch):
    entry_old = tf.gather_nd(self._storage, key_batch)
    entry_new = tf.scatter_nd(key_batch, -entry_old + value_batch,
                              self._storage.shape)
    self._storage = self._storage + entry_new

  def get_states(self):
    return self.all

  @staticmethod
  def from_states(*states):
    shape = tf.shape(states[0])
    s = BatchStorage(shape[0], shape[1], shape[2], dtype=states[0].dtype)
    s._storage = states[0]
    return s

  @property
  def all(self):
    return self._storage

  @property
  def size(self):
    return self._size

  @property
  def bsize(self):
    return self._bsize
