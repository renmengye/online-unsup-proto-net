"""Iterator for unsupervised episodes.

Potentially will include self-supervised augmentation.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.data.iterators.episode_iterator import EpisodeIterator
from fewshot.data.preprocessors.simclr_preprocessor import SIMCLRPreprocessor
from fewshot.data.registry import RegisterIterator

from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterIterator('unsupervised-augment-episode')
class UnsupervisedAugmentEpisodeIterator(EpisodeIterator):
  """Generates unsupervised episodes."""

  def __init__(self,
               dataset,
               config,
               sampler,
               batch_size,
               preprocessor=None,
               episode_processor=None,
               prefetch=True,
               **kwargs):
    # sampler.set_dataset(dataset)
    self._area_lb = config.area_lb
    self._color_distort = config.color_distort
    self._color_distort_strength = config.color_distort_strength
    self._flip = config.flip_left_right
    self._min_object_covered = config.min_object_covered
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
    s, flag, q = collection['support'], collection['flag'], collection['query']
    del collection['support']
    del collection['query']
    del collection['flag']
    dataset = self.dataset
    nclasses = self.nclasses
    img_s = dataset.get_images(s)
    lbl_s = np.array(collection['support_label'])
    del collection['support_label']
    T = self.maxlen

    # Note numpy does not give the desired behavior here.
    lbl_map, lbl_s = tf.unique(lbl_s)
    lbl_s_np = np.copy(lbl_s)
    lbl_s_masked = np.zeros_like(lbl_s_np) + self.unk_id
    lbl_s_gt = np.zeros_like(lbl_s_np) + self.unk_id

    N = img_s.shape[0]
    H = img_s.shape[1]
    W = img_s.shape[2]
    C = img_s.shape[3]
    simclr_proc = SIMCLRPreprocessor(
        H,
        W,
        color_distort=self._color_distort,
        flip=self._flip,
        color_distort_strength=self._color_distort_strength,
        min_object_covered=self._min_object_covered,
        area_range_lb=self._area_lb,
        num_views=self.config.num_aug,
        map_fn=self.config.map_fn)
    img_s = tf.image.convert_image_dtype(img_s, tf.float32)
    with tf.device("/cpu:0"):
      img_aug = simclr_proc.preprocess_batch(img_s)
    img_all = tf.concat([img_s, img_aug], axis=-1)  # [N, H, W, Cx3]

    epi = {
        'x_s': self.pad_x(img_all, T),
        'y_s': self.pad_y(lbl_s_masked, T),
        'y_gt': self.pad_y(lbl_s_gt, T),
        'y_full': self.pad_y(lbl_s, T),
        'flag_s': self.get_flag(lbl_s, T)
    }

    # For remaining additional info.
    for k in collection:
      epi[k] = self.pad_y(collection[k], T)

    if self.episode_processor is not None:
      epi = self.episode_processor(epi)
    return epi

  def _next(self):
    """Next example."""
    collection = self.sampler.sample_collection(self.nclasses, self.nquery,
                                                **self.kwargs)
    return self.process_one(collection)
