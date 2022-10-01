"""Iterator for regular mini-batches.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf


class EpochMinibatchIterator(object):
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
    self._allimages = dataset.get_images(dataset.get_ids())
    self._alllabels = dataset.get_labels(dataset.get_ids())
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
      idx = np.arange(self._allimages.shape[0])
      self._rnd.shuffle(idx)
      self._allimages = self._allimages[idx]
      self._alllabels = self._alllabels[idx]
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)

  def __iter__(self):
    return self._tf_dataset_iter

  def get_images(self, x):
    return {'x': x}

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    if self._totalrank == 1:
      images = self._allimages
      labels = self._alllabels
      # print('single machine')
    else:
      N = self._allimages.shape[0]
      nperrank = int(np.floor(N / self._totalrank))
      start = self._rank * nperrank
      end = (self._rank + 1) * nperrank
      images = self._allimages[start:end]
      labels = self._alllabels[start:end]
      # print(self._rank, start, end)
    ds = tf.data.Dataset.from_tensor_slices({'x': images, 'y': labels})
    # crop_size = self.dataset.get_images(0).shape[0]
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
