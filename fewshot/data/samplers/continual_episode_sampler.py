"""Sampler of hierarchical lifelong learning episodes.

It contains multiple sequences of episodes, softly blended together.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from fewshot.data.samplers.episode_sampler import EpisodeSampler
from fewshot.utils.logger import get as get_logger

log = get_logger()


class ContinualEpisodeSampler(EpisodeSampler):

  def __init__(self, subsampler, seed):
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
    super(ContinualEpisodeSampler, self).__init__(seed)
    self._subsampler = subsampler
    self._hierarchy_dict = None
    self._hierarchy_dict_keys = None
    self._stage_id = 0

  def set_dataset(self, dataset):
    """Hook the sampler with a dataset object."""
    super(ContinualEpisodeSampler, self).set_dataset(dataset)
    self._set_hierarchy_dict(dataset.get_hierarchy_dict())

  def set_stage(self, stage_id):
    self._stage_id = stage_id

  def _set_hierarchy_dict(self, hierarchy_dict):
    self._hierarchy_dict = hierarchy_dict

  def sample_episode_classes(self, n, max_num=-1, **kwargs):
    """Samples a sequence of classes relative to 0.

    Args:
      n: Int. Total number of classes.
      nstages: Int. Total number of stages.
      blender: String. Blender type.
      max_num: Int. Maximum number of examples.
      kwargs: Other parameters.
    """
    return self.subsampler.sample_episode_classes(n, max_num=max_num, **kwargs)

  def sample_classes(self, n, save_additional_info=False, **kwargs):
    """Samples a sequence of classes.

    Args:
      n: Int. Number of classes.
      kwargs: Other parameters.
    """
    episode_classes = self.sample_episode_classes(n, **kwargs)
    hmap = self.hierarchy_dict_keys
    results = []
    self.rnd.shuffle(self.hierarchy_dict[hmap[self._stage_id]])

    for c in episode_classes:
      results.append(self.hierarchy_dict[hmap[self._stage_id]][c])

    return results

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
