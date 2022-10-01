"""Sampler of lifelong learning episodes.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from fewshot.data.samplers.sampler import Sampler


class EpisodeSampler(Sampler):
  """Sampler interface."""

  def __init__(self, seed):
    """Initialize the sampler with a class dictionary, maps from class ID to a
    list of image IDs."""
    self._cls_dict = None
    self._cls_dict_keys = None
    self._dataset = None
    self._seed = seed
    self._rnd = np.random.RandomState(seed)

  def reset(self):
    """Reset randomness"""
    self._rnd = np.random.RandomState(self._seed)

  def set_dataset(self, dataset):
    """Hook the sampler with a dataset object."""
    self._set_cls_dict(dataset.get_cls_dict())
    self._dataset = dataset

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

  def sample_classes(self, n, **kwargs):
    """Samples a sequence of classes.

    Args:
      n: Int. Number of classes.
      kwargs: Other parameters.
    """
    episode_classes = self.sample_episode_classes(n, **kwargs)
    episode_classes = np.array(episode_classes)
    classmap = self.cls_dict_keys

    if 'celeb' in self._dataset.folder:
      shuffleidx = np.copy(classmap)
      self.rnd.shuffle(shuffleidx)
      u, c = np.unique(episode_classes, return_counts=True)
      index = -1
      classmap_ = []
      for u_, c_ in zip(u, c):
        success = False
        while not success:
          index += 1
          if len(self.cls_dict[shuffleidx[index]]) >= c_:
            # print(shuffleidx[index], len(self.cls_dict[shuffleidx[index]]), c_)
            success = True
        classmap_.append(shuffleidx[index])
      classmap_ = np.array(classmap_)
    else:
      classmap_ = self.rnd.choice(
          classmap, episode_classes.max() + 1, replace=False)
    return classmap_[episode_classes]

  def sample_episode_classes(self, n, **kwargs):
    """Samples a sequence of classes relative to 0.

    Args:
      n: Int. Number of classes.
      kwargs: Other parameters.
    """
    raise NotImplementedError()

  def sample_test_images(self, cls, image_ids, m):
    """Samples a sequence of query set. Making sure that newly sampled images
    will not overlap with the already sampled ones.

    Args:
      cls: List. List of class IDs.
      image_ids: List. List of image IDs already sampled.
      m: Int. number of images per class.

    Returns:
      A list of n*m class IDs, each class gets m query images.
    """
    result = []
    image_ids_set = set(image_ids)
    for c in cls:
      c_ = c % len(self.cls_dict_keys)
      k = self.cls_dict_keys[c_]
      all_images = self.cls_dict[k]
      all_images_set = set(all_images)
      remain = np.array(list(all_images_set.difference(image_ids_set)))
      result.extend(self.rnd.choice(remain, size=m, replace=False))
    return result

  def sample_images(self, classes, allow_repeat=False):
    """Samples images based on a sequence of classes.

    Args:
      classes: List. List of class ID (integer).
      allow_repeat: Bool. Whether images can repeat, default False.

    Returns:
      A list of image ID (integer).
    """
    image_map = {}
    image_ids = []
    counter = {}
    if 'celeb' not in self._dataset.folder:
      N = len(self.cls_dict_keys)
    for c in classes:
      if 'celeb' not in self._dataset.folder:
        c_ = c % N
      else:
        c_ = c

      if 'celeb' not in self._dataset.folder:
        k = self.cls_dict_keys[c_]
      else:
        k = c_
      if not allow_repeat:
        if c not in image_map:
          img_map = np.arange(len(self.cls_dict[k]))
          self.rnd.shuffle(img_map)
          image_map[c] = img_map
          counter[c] = 0
        image_ids.append(self.cls_dict[k][image_map[c][counter[c]]])
        counter[c] += 1
      else:
        rnd_idx = int(np.floor(self.rnd.uniform(0, len(self.cls_dict[k]))))
        image_ids.append(self.cls_dict[k][rnd_idx])
    return image_ids

  def sample_collection(self,
                        n,
                        m,
                        allow_repeat=False,
                        save_additional_info=False,
                        **kwargs):
    """Samples an episode of image IDs.

    Args:
      n: Int. Number of classes.
      m: Int. Number of query images per class.
      allow_repeat: Bool. Whether allow images to repeat.
      save_additional_info: Bool. Whether to add additional episodic
        information.
      kwargs: Dict. Any other parameters needed for sampling support data.

    Returns:
      A tuple of
        - a list of support image IDs, and
        - a list of query image IDs.
    """
    if save_additional_info:
      results = self.sample_classes(n, save_additional_info=True, **kwargs)
      if isinstance(results, tuple):
        cls_support = results[0]
        info = results[1]
      else:
        cls_support = results
        info = {}
    else:
      cls_support = self.sample_classes(n, **kwargs)
    cls_unique, idx = np.unique(np.array(cls_support), return_index=True)
    support = self.sample_images(cls_support, allow_repeat=allow_repeat)
    query = self.sample_test_images(cls_unique, support, m)
    collection = {
        'support': support,
        'support_label': cls_support,
        'query': query,
        'query_label': np.repeat(cls_unique, m)
    }

    if save_additional_info:
      for k in info:
        collection[k] = info[k]
      if 'stage_id' in info and m > 0:
        collection['stage_id_q'] = np.repeat(
            np.array(info['stage_id'])[idx], m)

    return collection

  @property
  def rnd(self):
    return self._rnd

  @property
  def cls_dict(self):
    assert self._cls_dict is not None, 'Uninitialized cls_dict'
    return self._cls_dict

  @property
  def cls_set_dict(self):
    assert self._cls_set_dict is not None, 'Uninitialized cls_set_dict'
    return self._cls_set_dict
