"""Iterator for unsupervised episodes for video datasets like SAY.

Potentially will include self-supervised augmentation.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

import tensorflow as tf

# from fewshot.data.iterators.sim_episode_iterator import SimEpisodeIterator
from fewshot.data.iterators.iterator import Iterator
from fewshot.data.preprocessors.simclr_preprocessor import SIMCLRPreprocessor
from fewshot.data.registry import RegisterIterator


@RegisterIterator('unsupervised-augment-vid-episode')
class UnsupervisedAugmentVidEpisodeIterator(Iterator):
  """Generates unsupervised episodes."""

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
               **kwargs):
    self._dataset = dataset
    self._config = config
    self._preprocessor = preprocessor
    self._data_format2 = data_format
    self._batch_size = batch_size
    self._rnd = np.random.RandomState(seed)
    self._rank = rank
    self._totalrank = totalrank
    self._simclr_proc = SIMCLRPreprocessor(
        224,
        224,
        color_distort=config.color_distort,
        flip=config.flip_left_right,
        color_distort_strength=config.color_distort_strength,
        area_range_lb=config.area_lb,
        rotate=config.random_rotate,
        rotate_degrees=config.random_rotate_angle,
        gaussian_blur=config.gaussian_blur,
        gaussian_blur_prob=config.gaussian_blur_prob,
        motion_blur=config.motion_blur,
        motion_blur_prob=config.motion_blur_prob,
        num_views=config.num_aug,
        min_object_covered=config.min_object_covered,
        map_fn=config.map_fn)
    self._prefetch = prefetch
    self._tf_dataset = self.get_dataset()
    self._tf_dataset_iter = iter(self._tf_dataset)

  def process_one(self, idx):
    """Process one episode."""
    # with tf.device("/cpu:0"):
    episode = self.dataset.get_episode(
        idx, length=self.config.maxlen, interval=self.config.frame_rate)

    if episode is not None:
      # Truncate according to rank.
      M = self._totalrank
      T = episode['x'].shape[0]
      m = T // M
      start2 = self._rank * m
      end2 = (self._rank + 1) * m
      episode['x'] = episode['x'][start2:end2]

      x = episode['x']
      x = self.preprocessor(x)
      x_aug = self._simclr_proc.preprocess_batch(x)
      if self.config.num_aug >= 2:
        x_all = x_aug  # [N, H, W, Cx2]
      else:
        x_all = tf.concat([x, x_aug], axis=-1)  # [N, H, W, Cx3]

      if self._data_format2 == "NCHW":
        x_all = tf.transpose(x_all, [0, 3, 1, 2])
      epi = {
          'x_s': x_all,
          'y_s': np.zeros([x_all.shape[0]]) + self.config.unk_id,
          'y_gt': np.zeros([x_all.shape[0]]) + self.config.unk_id,
          'y_full': np.zeros([x_all.shape[0]]) + self.config.unk_id,
          'flag_s': np.ones([x_all.shape[0]]),
          'id': tf.zeros([x_all.shape[0]], dtype=tf.int32) + idx
      }
      return epi
    else:
      return None

  def __iter__(self):
    return self._tf_dataset_iter

  def get_generator(self):
    """Gets generator function, for tensorflow Dataset object."""
    while True:
      # Let's not shuffle here, truly online.
      self._idx = np.arange(len(self.dataset))
      self._reset_flag = False
      T = self.config.frame_rate * self.config.maxlen
      rnd_start = self._rnd.randint(0, T)
      self._idx = self._idx[rnd_start::T]
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
    ds = ds.batch(self.batch_size)
    if self._prefetch:
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  def reset(self):
    self._reset_flag = True

  @property
  def dataset(self):
    """Dataset source."""
    return self._dataset

  @property
  def preprocessor(self):
    """Image preprocessor."""
    return self._preprocessor

  @property
  def batch_size(self):
    """Number of episodes."""
    return self._batch_size

  @property
  def tf_dataset(self):
    return self._tf_dataset

  @property
  def config(self):
    """Config"""
    return self._config
