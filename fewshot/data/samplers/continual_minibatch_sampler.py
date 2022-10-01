"""Sampler of hierarchical lifelong learning episodes.

It contains multiple sequences of episodes, softly blended together.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from fewshot.data.registry import RegisterSampler
from fewshot.data.samplers.sampler import Sampler
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterSampler('continual_minibatch')
class ContinualMinibatchSampler(Sampler):

  def __init__(self, seed):
    """Initialize a hiarchical episode sampler.

    Args:
      subsampler: Sampler that samples an individual episode.
      blender: A blender that blends a few sequential episode.
      use_class_hierarchy: Whether to use the class hierarchy defined in the
        dataset.
      use_same_family: Whether to use the same class family across different
        context environment.
      seed: Int. Random seed.
    """
    self._hierarchy_dict = None
    self._hierarchy_dict_keys = None
    self._stage_id = 0
    self._cls_dict = None
    self._dataset = None
    self._seed = seed
    self._rnd = np.random.RandomState(seed)

  def set_dataset(self, dataset):
    """Hook the sampler with a dataset object."""
    self._set_cls_dict(dataset.get_cls_dict())
    self._dataset = dataset
    self._set_hierarchy_dict(dataset.get_hierarchy_dict())
    self._hierarchy_img_dict = self.get_hierarchy_img_dict()

  def set_stage(self, stage_id):
    self._stage_id = stage_id

  def _set_hierarchy_dict(self, hierarchy_dict):
    self._hierarchy_dict = hierarchy_dict

  def get_hierarchy_img_dict(self):
    results = {}
    for k in self.hierarchy_dict:
      results[k] = []
      for c in self.hierarchy_dict[k]:
        results[k].extend(self.cls_dict[c])
    return results

  def _set_cls_dict(self, cls_dict):
    assert self._cls_dict is None, 'Class dict can only be set once.'
    self._cls_dict = cls_dict
    all_image_ids_set = set()
    msg = 'An image cannot exist in both classes'
    klist = cls_dict.keys()
    for cls in klist:
      image_ids = cls_dict[cls]
      for i in image_ids:
        # Check class is non-overlap.
        assert i not in all_image_ids_set, msg
        all_image_ids_set.add(i)

    self._cls_set_dict = {}
    for cls in klist:
      self._cls_set_dict[cls] = set(cls_dict[cls])

  @property
  def cls_dict_keys(self):
    if self._cls_dict_keys is None:
      self._cls_dict_keys = list(self.cls_dict.keys())
    return self._cls_dict_keys

  def sample_collection(self, batch_size):
    family = self.hierarchy_dict_keys[self._stage_id]
    idx = self.hierarchy_img_dict[family]
    self._rnd.shuffle(idx)
    return idx[:batch_size]

  @property
  def subsampler(self):
    """Sampler for each phase."""
    return self._subsampler

  @property
  def hierarchy_dict_keys(self):
    if self._hierarchy_dict_keys is None:
      self._hierarchy_dict_keys = list(self.hierarchy_dict.keys())
    return self._hierarchy_dict_keys

  @property
  def hierarchy_dict(self):
    assert self._hierarchy_dict is not None, 'Uninitialized hierarchy_dict'
    return self._hierarchy_dict

  @property
  def cls_dict(self):
    assert self._cls_dict is not None, 'Uninitialized cls_dict'
    return self._cls_dict

  @property
  def cls_set_dict(self):
    assert self._cls_set_dict is not None, 'Uninitialized cls_set_dict'
    return self._cls_set_dict

  @property
  def hierarchy_img_dict(self):
    assert self._hierarchy_img_dict is not None, 'Uninitialized hierarchy_img_dict'  # NOQA
    return self._hierarchy_img_dict
