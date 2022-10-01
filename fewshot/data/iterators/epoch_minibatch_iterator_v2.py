"""Iterator for regular mini-batches.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf


class EpochMinibatchIteratorV2(object):
  """Generates mini-batches for pretraining."""

  def __init__(self,
               dataset,
               batch_size,
               prefetch=True,
               shuffle=True,
               preprocessor=None,
               rank=0,
               totalrank=1,
               drop_remainder=False,
               seed=0,
               **kwargs):
    self._dataset = dataset
    self._preprocessor = preprocessor
    assert preprocessor is not None
    assert batch_size > 0, 'Need a positive number for batch size'
    self._batch_size = batch_size
    self._prefetch = prefetch
    self._drop_remainder = drop_remainder
    self._rank = rank
    self._totalrank = totalrank
    self._idx = self.dataset.get_ids()
    self._shuffle = shuffle
    self._rnd = np.random.RandomState(seed)
    self.reset()

  def __len__(self):
    N = self.dataset.get_size()
    size = np.floor(N / self._totalrank) / float(self.batch_size)
    if self._drop_remainder:
      return int(np.floor(size))
    else:
      return int(np.ceil(size))

  def reset(self):
    if self._shuffle:
      self._rnd.shuffle(self._idx)
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)

  def __iter__(self):
    return self._tf_dataset_iter

  def get_images(self, x):
    return {'x': x}

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    N = len(self._idx)
    if self._totalrank == 1:
      start = 0
      end = N
    else:
      nperrank = int(np.floor(N / self._totalrank))
      start = self._rank * nperrank
      end = (self._rank + 1) * nperrank

    def get_images_and_labels(x):
      x = int(x)
      return (self.dataset.get_images(x), self.dataset.get_labels(x))

    def map_fn(x):
      x, y = tf.py_function(get_images_and_labels, [x], [tf.uint8, tf.int64])
      return {'x': x, 'y': y}

    ds = tf.data.Dataset.from_tensor_slices(self._idx[start:end])
    ds = ds.map(map_fn)
    ds = ds.batch(self.batch_size, drop_remainder=self.drop_remainder)

    def preprocess(data):
      data['x'] = self.preprocessor(data['x'])
      return data

    if self.preprocessor is not None:
      ds = ds.map(preprocess)

    if self._prefetch:
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  @property
  def dataset(self):
    """Dataset object."""
    return self._dataset

  @property
  def preprocessor(self):
    """Data preprocessor."""
    return self._preprocessor

  @property
  def batch_size(self):
    """Batch size."""
    return self._batch_size

  @property
  def tf_dataset(self):
    return self._tf_dataset

  @property
  def drop_remainder(self):
    return self._drop_remainder
