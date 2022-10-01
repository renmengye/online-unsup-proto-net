"""SAYCam dataset API.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# import glob
import os

import numpy as np
import cv2
from tqdm import tqdm

from fewshot.data.datasets.pickle_cache_dataset import PickleCacheDataset
from fewshot.data.registry import RegisterDataset


@RegisterDataset("say-cam-labeled")
class SAYCamLabeledDataset(PickleCacheDataset):

  def __init__(self, folder, split, seed=0):
    self._seed = seed
    super(SAYCamLabeledDataset, self).__init__(folder, split)
    train_test_ratio = 0.5
    subsample = 0.1
    # subsample = 1.0
    self._rnd = np.random.RandomState(seed)
    self._split_idx = self._create_split(train_test_ratio, subsample)

  def _create_split(self, ratio, subsample):
    """Creates training test split.
    TODO: ask Emin for the original split.
    """
    num_ex = int(np.floor(subsample * len(self._images) * ratio))
    idx = np.arange(len(self._images))
    self._rnd.shuffle(idx)
    print(self._split, num_ex)
    # if subsample < 1.0:
    #   L = int(subsample * len(self._images))
    # else:
    #   L = num_ex
    if self._split in ['train']:
      return idx[:num_ex]
    else:
      return idx[num_ex:2 * num_ex]

  def get_cache_path(self):
    """Gets cache file name."""
    cache_path = os.path.join(self._folder, 'cache.pkl')
    return cache_path

  def _read_dataset(self):
    label_idx = []
    label_str = []
    images = []
    folder = self._folder
    label_str = os.listdir(folder)
    label_str = list(
        sorted(
            filter(lambda x: os.path.isdir(os.path.join(folder, x)),
                   label_str)))
    print(folder, label_str)
    for label, subfolder in tqdm(enumerate(label_str)):
      subfolder_ = os.path.join(folder, subfolder)
      img_list = os.listdir(subfolder_)
      for img_fname in img_list:
        fname_ = os.path.join(subfolder_, img_fname)
        img = cv2.imread(fname_)
        img_str = cv2.imencode(".jpg", img)[1]
        images.append(img_str)
        label_idx.append(label)
    label_idx = np.array(label_idx)
    data = {'images': images, 'labels': label_idx, 'label_str': label_str}
    print(images, len(images), label_idx, len(label_idx), label_str)
    return data

  def get_images(self, inds):
    images = None
    if type(inds) == int:
      images = cv2.imdecode(self.images[inds], 1)
    else:
      for ii, item in enumerate(inds):
        im = cv2.imdecode(self.images[item], 1)
        if images is None:
          images = np.zeros([len(inds), im.shape[0], im.shape[1], im.shape[2]],
                            dtype=im.dtype)
        images[ii] = im
    return images

  def get_labels(self, inds):
    """Gets the label of an image or a batch of images."""
    return self.labels[inds]

  def get_ids(self):
    return self._split_idx

  def get_size(self):
    """Get the total number of images."""
    return len(self._split_idx)

  def __len__(self):
    return len(self._split_idx)

  @property
  def folder(self):
    """Data folder."""
    return self._folder

  @property
  def split(self):
    """Data split."""
    return self._split


if __name__ == '__main__':
  # SAYCamLabeledDataset('./data/say-cam-labeled/S_labeled', 'train')
  d = SAYCamLabeledDataset('/mnt/research/datasets/say-cam-labeled/S_labeled',
                           'train')
  print(len(d))
