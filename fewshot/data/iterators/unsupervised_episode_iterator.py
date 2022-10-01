"""Iterator for unsupervised episodes.

Potentially will include self-supervised augmentation.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.data.iterators.episode_iterator import EpisodeIterator
from fewshot.data.registry import RegisterIterator


@RegisterIterator('unsupervised-episode')
class UnsupervisedEpisodeIterator(EpisodeIterator):
  """Generates unsupervised episodes."""

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

    epi = {
        'x_s': self.pad_x(img_s, T),
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
