"""Iterator for regular mini-batches.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
from fewshot.data.preprocessors import SIMCLRPreprocessor


class SIMCLRAugMinibatchIterator(object):
  """Generates mini-batches for pretraining."""

  def __init__(self,
               dataset,
               batch_size,
               prefetch=True,
               preprocessor=None,
               rank=0,
               totalrank=1,
               drop_remainder=False,
               flip=True,
               color_distort=True,
               color_distort_strength=1.0,
               area_range_lb=0.08,
               min_object_covered=0.1,
               num_views=2,
               cycle=False,
               seed=0):
    self._dataset = dataset
    self._preprocessor = preprocessor
    assert preprocessor is not None
    assert batch_size > 0, 'Need a positive number for batch size'
    self._batch_size = batch_size
    self._prefetch = prefetch
    self._drop_remainder = drop_remainder
    self._rank = rank
    self._totalrank = totalrank
    self._cycle = cycle
    self._color_distort = color_distort
    self._color_distort_strength = color_distort_strength
    self._flip = flip
    self._area_range_lb = area_range_lb
    self._min_object_covered = min_object_covered
    self._num_views = num_views
    self._allimages = dataset.get_images(dataset.get_ids())
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

  def reset(self):
    self._rnd.shuffle(self._allimages)
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
    else:
      N = self._allimages.shape[0]
      nperrank = int(np.floor(N / self._totalrank))
      start = self._rank * nperrank
      end = (self._rank + 1) * nperrank
      images = self._allimages[start:end]
    ds = tf.data.Dataset.from_tensor_slices({'x': images})
    if self._cycle:
      ds = ds.repeat().shuffle(images.shape[0] // 5)
    img = self.dataset.get_images(0)
    crop_size_h = img.shape[0]
    crop_size_w = img.shape[1]
    simclr_preprocessor = SIMCLRPreprocessor(
        crop_size_h,
        crop_size_w,
        color_distort=self._color_distort,
        min_object_covered=self._min_object_covered,
        flip=self._flip,
        color_distort_strength=self._color_distort_strength,
        area_range_lb=self._area_range_lb,
        num_views=self._num_views)

    def simclr_preprocess(data):
      x_aug = simclr_preprocessor(data['x'])
      x_orig = tf.image.convert_image_dtype(data['x'], tf.float32)
      data['x'] = tf.concat([x_orig, x_aug], axis=-1)
      return data

    ds = ds.map(simclr_preprocess)
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
  def sampler(self):
    """Mini-batch sampler."""
    return self._sampler

  @property
  def step(self):
    """Number of steps."""
    return self.sampler._step

  @property
  def epoch(self):
    """Number of epochs."""
    return self.sampler._epoch

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
