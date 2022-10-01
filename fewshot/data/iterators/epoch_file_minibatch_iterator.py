"""Iterator for regular mini-batches.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from fewshot.data.datasets.say_cam_time import SAYCamTimeDataset

import cv2
import numpy as np
import tensorflow as tf
import h5py

from fewshot.utils.logger import get as get_logger

log = get_logger()


def _make_iter(img_arr, l_arr, start=0, end=None, interval=1):
  """Makes an PNG encoding string iterator."""
  prev = 0
  if end is None:
    end = len(l_arr)
  l_cum = np.cumsum(l_arr)
  for i in range(start, end, interval):
    idx = l_cum[i]
    if i > 0:
      prev = l_cum[i - 1]
    else:
      prev = 0
    yield cv2.imdecode(img_arr[prev:idx], -1)


class EpochFileMinibatchIterator(object):
  """Generates mini-batches from chunks of files."""

  def __init__(self,
               dataset,
               batch_size,
               shuffle=True,
               frame_rate=1,
               prefetch=True,
               preprocessor=None,
               rank=0,
               totalrank=1,
               drop_remainder=False,
               seed=0):
    self._dataset = dataset
    self._preprocessor = preprocessor
    # assert preprocessor is not None
    assert batch_size > 0, 'Need a positive number for batch size'
    self._batch_size = batch_size
    self._prefetch = prefetch
    self._drop_remainder = drop_remainder
    self._rank = rank
    self._totalrank = totalrank
    self._filenames = self.dataset.get_filenames()
    self._shuffle = shuffle
    self._frame_rate = frame_rate
    log.info('Shuffle: {}'.format(shuffle))
    log.info('File iterator frame rate: {}'.format(frame_rate))
    self._rnd = np.random.RandomState(seed)
    self.reset()
    # self._tf_dataset = self.get_dataset()
    # self._tf_dataset_iter = iter(self._tf_dataset)

  def __len__(self):
    N = self.dataset.get_size()
    size = np.floor(N / self._totalrank) / float(self.batch_size)
    if self._drop_remainder:
      return int(np.floor(size))
    else:
      return int(np.ceil(size))

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""

    if self._totalrank > 1:
      N = len(self._filenames)
      nperrank = int(np.floor(N / self._totalrank))
      start = self._rank * nperrank
      end = (self._rank + 1) * nperrank
      filenames = self._filenames[start:end]
    else:
      filenames = self._filenames

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dtype_dict = {
        'x': tf.uint8,
        'y': tf.int64,
    }
    shape_dict = {
        'x': tf.TensorShape([None, None, None]),
        'y': tf.TensorShape([]),
    }

    def generator(filename):
      # print(filename)
      with h5py.File(filename, 'r') as f:
        images = f['images'][:]
        images_len = f['images_len'][:]
        labels = f['labels'][:]
      it = _make_iter(images, images_len, 0, None, interval=self._frame_rate)
      for i, l in zip(it, labels):
        yield {'x': i, 'y': l}

    def interleave_fn(filename):
      return tf.data.Dataset.from_generator(
          generator, dtype_dict, shape_dict, args=(filename,))

    dataset = dataset.interleave(
        interleave_fn, cycle_length=8, block_length=16).batch(
            self.batch_size, drop_remainder=self.drop_remainder)

    def preprocess(data):
      data['x'] = self.preprocessor(data['x'])
      return data

    if self.preprocessor is not None:
      dataset = dataset.map(preprocess)

    if self._prefetch:
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

  def __iter__(self):
    return self._tf_dataset_iter

  def reset(self):
    if self._shuffle:
      self._rnd.shuffle(self._filenames)
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)

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


if __name__ == '__main__':
  it = EpochFileMinibatchIterator(
      SAYCamTimeDataset("/mnt/research/datasets/say-cam/h5_data_shuffle", "S"),
      256)

  for i in it:
    print(i)
    break
