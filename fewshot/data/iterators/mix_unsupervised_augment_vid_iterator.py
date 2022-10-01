"""Iterators for few-shot episode.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

import tensorflow as tf

from fewshot.data.iterators.unsupervised_augment_vid_episode_iterator import UnsupervisedAugmentVidEpisodeIterator  # NOQA
from fewshot.data.preprocessors.simclr_preprocessor import SIMCLRPreprocessor
from fewshot.data.registry import RegisterIterator


@RegisterIterator('mix-unsupervised-augment-vid-noniid')
class MixUnsupAugmentVidIterator(UnsupervisedAugmentVidEpisodeIterator):

  def __init__(self,
               dataset,
               config,
               batch_size,
               preprocessor=None,
               data_format="NCHW",
               prefetch=True,
               seed=0,
               rank=0,
               totalrank=1,
               drop_remainder=False,
               **kwargs):
    self._dataset = dataset
    self._config = config
    self._maxlen = config.maxlen
    self._reset_flag = False
    assert config.maxlen > 0
    self._unk_id = config.unk_id
    self._preprocessor = preprocessor
    self._data_format2 = data_format
    self._batch_size = batch_size
    self._rnd = np.random.RandomState(seed)
    self._rank = rank
    self._totalrank = totalrank
    self._drop_remainder = drop_remainder
    self._prefetch = prefetch
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)

  def process_one(self, i):
    item = self.dataset.get_episode(
        i, length=self.config.maxlen, interval=self.config.frame_rate)
    if item is not None:
      item['id'] = np.zeros([item['x'].shape[0]], dtype=np.int64) + i
    return item

  def get_generator(self):
    """Gets generator function, for tensorflow Dataset object."""
    while True:
      self._idx = np.arange(len(self.dataset))
      # IID reader, shuffle here.
      self._rnd.shuffle(self._idx)
      N = len(self._idx)
      nperrank = int(np.floor(N / self._totalrank))
      start = self._rank * nperrank
      end = (self._rank + 1) * nperrank
      self._idx = self._idx[start:end]
      self._reset_flag = False

      for i in self._idx:
        if not self._reset_flag:
          item = self.process_one(i)
          if item is not None:
            yield item
          else:
            break
        else:
          break

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    dummy = self.process_one(0)
    dtype_dict = dict([(k, dummy[k].dtype) for k in dummy])
    shape_dict = dict([(k, tf.shape(dummy[k])) for k in dummy])
    ds = tf.data.Dataset.from_generator(self.get_generator, dtype_dict,
                                        shape_dict)
    ds = ds.interleave(
        tf.data.Dataset.from_tensor_slices,
        cycle_length=20,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            buffer_size=4000)

    img = dummy['x']
    crop_size_h = img.shape[1]
    crop_size_w = img.shape[2]
    simclr_preprocessor = SIMCLRPreprocessor(
        crop_size_h,
        crop_size_w,
        color_distort=self.config.color_distort,
        flip=self.config.flip_left_right,
        color_distort_strength=self.config.color_distort_strength,
        area_range_lb=self.config.area_lb,
        min_object_covered=self.config.min_object_covered,
        num_views=self.config.num_aug)

    def simclr_preprocess(data):
      x = simclr_preprocessor(data['x'], bbox=None)
      return {'x': x, 'id': data['id']}

    def preprocess(data):
      x = self.preprocessor(data['x'])
      return {'x': x, 'id': data['id']}

    ds = ds.map(simclr_preprocess, deterministic=True)
    ds = ds.batch(self.batch_size)
    ds = ds.map(preprocess, deterministic=True)
    if self._prefetch:
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  def reset(self):
    self._reset_flag = True

  @property
  def config(self):
    """Config."""
    return self._config

  def __len__(self):
    N = self.dataset.get_size()
    size = np.floor(N / self._totalrank) / float(self.batch_size)
    size /= self.config.frame_rate
    if self._drop_remainder:
      return int(np.floor(size))
    else:
      return int(np.ceil(size))
