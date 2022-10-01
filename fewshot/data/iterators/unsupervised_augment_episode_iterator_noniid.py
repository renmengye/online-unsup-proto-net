"""Unsupervised iterator with non-iid.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from fewshot.data.preprocessors.simclr_preprocessor import SIMCLRPreprocessor
from fewshot.data.iterators.unsupervised_augment_episode_iterator import UnsupervisedAugmentEpisodeIterator  # NOQA
import tensorflow as tf
from fewshot.data.registry import RegisterIterator
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterIterator('unsupervised-augment-noniid')
class UnsupervisedAugmentEpisodeIteratorNonIID(
    UnsupervisedAugmentEpisodeIterator):  # NOQA

  def __init__(self,
               dataset,
               config,
               sampler,
               batch_size,
               preprocessor=None,
               rank=0,
               totalrank=1,
               drop_remainder=False,
               episode_processor=None,
               prefetch=True,
               **kwargs):
    # sampler.set_dataset(dataset)
    # assert False, 'not used'
    self._area_lb = config.area_lb
    self._color_distort = config.color_distort
    self._color_distort_strength = config.color_distort_strength
    self._flip = config.flip_left_right
    self._min_object_covered = config.min_object_covered
    self._rank = rank
    self._totalrank = totalrank
    self._drop_remainder = drop_remainder
    if config.map_fn:
      log.info('Using map fn to do data augmentation')
    else:
      log.info('Not using map fn to do data augmentation')
    super(UnsupervisedAugmentEpisodeIterator, self).__init__(
        dataset,
        config,
        sampler,
        batch_size,
        preprocessor=preprocessor,
        episode_processor=episode_processor,
        prefetch=prefetch,
        **kwargs)

  def process_one(self, collection):
    """Process one episode.

    Args:
      Collection dictionary that contains the following keys:
        support: np.ndarray. Image ID in the support set.
        flag: np.ndarray. Binary flag indicating whether it is labeled (1) or
          unlabeled (0).
        query: np.ndarray. Image ID in the query set.
    """
    s = collection['support']
    dataset = self.dataset
    img_s = dataset.get_images(s)
    img_s = tf.image.convert_image_dtype(img_s, tf.float32)
    epi = {'x': img_s}
    return epi

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    dummy = self._next()
    self.sampler.reset()
    dtype_dict = dict([(k, dummy[k].dtype) for k in dummy])
    shape_dict = dict(
        [(k, [None] + list(tf.shape(dummy[k])[1:])) for k in dummy])
    ds = tf.data.Dataset.from_generator(self.get_generator, dtype_dict,
                                        shape_dict)
    ds = ds.interleave(tf.data.Dataset.from_tensor_slices, cycle_length=1)
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
      return {'x': x}

    def preprocess(data):
      data['x'] = self.preprocessor(data['x'])
      return data

    ds = ds.map(simclr_preprocess)
    ds = ds.batch(self.batch_size)
    if self.preprocessor is not None:
      ds = ds.map(preprocess)
    # if self._prefetch:
    #   ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  def __len__(self):
    N = self.dataset.get_size()
    size = np.floor(N / self._totalrank) / float(self.batch_size)
    if self._drop_remainder:
      return int(np.floor(size))
    else:
      return int(np.ceil(size))