"""Iterator for unsupervised episodes.

Potentially will include self-supervised augmentation.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.data.iterators.sim_episode_iterator import SimEpisodeIterator
from fewshot.data.preprocessors.simclr_preprocessor import SIMCLRPreprocessor
from fewshot.data.registry import RegisterIterator


@RegisterIterator('unsupervised-augment-sim-episode')
class UnsupervisedAugmentSimEpisodeIterator(SimEpisodeIterator):
  """Generates unsupervised episodes."""

  def __init__(self,
               dataset,
               config,
               sampler,
               batch_size,
               data_format="NCHW",
               **kwargs):
    self._area_lb = config.area_lb
    self._color_distort = config.color_distort
    self._color_distort_strength = config.color_distort_strength
    self._min_object_covered = config.min_object_covered
    self._flip = config.flip_left_right
    self._unk_id = config.unk_id
    self._num_aug = config.num_aug
    self._map_fn = config.map_fn
    self._data_format2 = data_format
    self._simclr_proc = SIMCLRPreprocessor(
        120,
        160,
        color_distort=self._color_distort,
        flip=self._flip,
        color_distort_strength=self._color_distort_strength,
        area_range_lb=self._area_lb,
        num_views=self._num_aug,
        min_object_covered=self._min_object_covered,
        map_fn=self._map_fn)
    super(UnsupervisedAugmentSimEpisodeIterator, self).__init__(
        dataset, config, sampler, batch_size, data_format="NHWC", **kwargs)

  def process_one(self, collection):
    """Process one episode.

    Args:
      Collection dictionary that contains the following keys:
        support: np.ndarray. Image ID in the support set.
        flag: np.ndarray. Binary flag indicating whether it is labeled (1) or
          unlabeled (0).
        query: np.ndarray. Image ID in the query set.
    """
    episode = super(UnsupervisedAugmentSimEpisodeIterator,
                    self).process_one(collection)
    if episode is None:
      return None
    x_s = episode['x_s']
    x_att = episode['x_att']
    img_s = tf.concat([x_s, x_att], axis=-1)
    N = img_s.shape[0]
    H = img_s.shape[1]
    W = img_s.shape[2]
    C = img_s.shape[3]

    img_s = tf.image.convert_image_dtype(img_s, tf.float32)

    import numpy as np
    B = x_att.shape[0]
    bbox = np.zeros([B, 4], dtype=np.float32)

    for i in range(B):
      nonzero = np.nonzero(x_att[i])
      if len(nonzero[0]) > 0:
        bbox_ymin = float(nonzero[0].min()) / float(H)
        bbox_ymax = float(nonzero[0].max()) / float(H)
        bbox_xmin = float(nonzero[1].min()) / float(W)
        bbox_xmax = float(nonzero[1].max()) / float(W)
        bbox[i, 0] = bbox_ymin
        bbox[i, 1] = bbox_xmin
        bbox[i, 2] = bbox_ymax
        bbox[i, 3] = bbox_xmax
      else:
        bbox[i, 0] = 0.0
        bbox[i, 1] = 0.0
        bbox[i, 2] = 1.0
        bbox[i, 3] = 1.0

    img_aug = self._simclr_proc.preprocess_batch(img_s, bbox=bbox)
    img_all = tf.concat([img_s, img_aug], axis=-1)  # [N, H, W, Cx3]

    if self._data_format2 == "NCHW":
      img_all = tf.transpose(img_all, [0, 3, 1, 2])

    # tf.print('all', img_all.shape)
    epi = {
        'x_s': img_all,
        'y_s': tf.zeros_like(episode['y_s']) + self._unk_id,
        'y_gt': tf.zeros_like(episode['y_gt']) + self._unk_id,
        'y_full': episode['y_full'],
        'flag_s': episode['flag_s']
    }
    return epi

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    dummy = self._next()
    self.sampler.reset()
    dtype_dict = dict([(k, dummy[k].dtype) for k in dummy])
    shape_dict = dict([(k, tf.shape(dummy[k])) for k in dummy])
    ds = tf.data.Dataset.from_generator(self.get_generator, dtype_dict,
                                        shape_dict)
    ds = ds.batch(self.batch_size)
    if self._prefetch:
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
