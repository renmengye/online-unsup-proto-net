"""CelebA dataset.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import os
import numpy as np
import pickle as pkl
from fewshot.data.datasets.pickle_cache_dataset import PickleCacheDataset
from fewshot.data.registry import RegisterDataset
from fewshot.utils.logger import get as get_logger
from tqdm import tqdm

import pandas as pd

log = get_logger()


@RegisterDataset('celeb-a-identity')
class CelebAIdentityDataset(PickleCacheDataset):

  def __init__(self, folder, split, image_db=None):
    """Creates a dataset with pickle cache.

    Args:
      folder: String. Folder path.
      split: String. Split name.
    """
    assert folder is not None
    assert split is not None
    self._folder = folder
    self._split = split
    if image_db is None:
      allimage = os.path.join(folder, 'allimage.npy')
      self._images = np.load(allimage)
      log.info('Reading from {}'.format(allimage))
    else:
      log.info('Reusing same image dict')
      self._images = image_db
    data = self.read_dataset()
    self._split_list = data['split_list']
    self._cls_dict = None
    self._labels = None
    identitylist = pd.read_csv(
        os.path.join(self.folder, 'identity_CelebA.txt'), sep=' ', header=None)
    print(self._images.shape)
    label = np.array(identitylist)[:, 1].astype(np.int64) - 1
    print(label)
    print(label.shape, label.max(), label.min())
    self._labels = label  # [:, self._sel]
    self._cls_dict = self.get_cls_dict()

  def _read_dataset(self):
    split_set = set()
    split_file = os.path.join(self.folder, self.split + '.txt')
    with open(split_file, 'r') as f:
      for i in f.readlines():
        split_set.add(int(i.strip('\n')))
    split_list = np.array(sorted(list(split_set)))
    return {'split_list': split_list}

  def read_cache(self, cache_path):
    """Reads dataset from cached pkl file.

    Args:
      cache_path: filename of the cache.

    Returns:
      dict: data dictionary, None if the cache doesn't exist.
    """
    if os.path.exists(cache_path):
      with open(cache_path, 'rb') as f:
        data = pkl.load(f)
      return data
    else:
      return None

  def get_size(self):
    """Gets the total number of images."""
    return len(self._split_list)

  def get_images(self, inds):
    """Get images."""
    return self.images[inds]

  def get_ids(self):
    return self._split_list

  def get_labels(self, inds):
    """Get labels for pretraining."""
    return self._labels[inds]

  def get_cls_dict(self):
    """Get attribute dictionary."""
    # Read the inverted index here.
    cls_dict = {}
    for i in self._split_list:
      ll = self.labels[i]
      if ll in cls_dict:
        cls_dict[ll].append(i)
      else:
        cls_dict[ll] = [i]
    # self._cls_dict = cls_dict
    return cls_dict


if __name__ == '__main__':
  for sp in ['val', 'test', 'train']:
    ds = CelebAIdentityDataset('/mnt/research/datasets/celeb_a', sp, None)
