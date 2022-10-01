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
import pickle as pkl

from fewshot.data.datasets.pickle_cache_dataset import PickleCacheDataset
from fewshot.data.registry import RegisterDataset


@RegisterDataset("say-cam-labeled-v2")
class SAYCamLabeledV2Dataset(PickleCacheDataset):

  def __init__(self, folder, split, seed=0):
    self._seed = seed
    super(SAYCamLabeledV2Dataset, self).__init__(folder, split)
    train_test_ratio = 0.5
    self._rnd = np.random.RandomState(seed)
    self._fname_list = self._data['fname_list']
    self._cls_dict = self.get_cls_dict()
    self._split_idx = self._create_split(train_test_ratio)

  def _create_split(self, ratio):
    """Creates training test split.
    TODO: ask Emin for the original split.
    """
    split = []
    fname_list = np.array(self._fname_list)
    for lbl in self._cls_dict:
      image_ids = np.array(self._cls_dict[lbl])
      fnames = fname_list[image_ids]
      # tup = [(f, i) for f, i in zip(fnames, image_ids)]
      # tup = np.stack([fnames, image_ids], axis=1)
      sortidx = np.argsort(fnames)
      sorted_image_ids = image_ids[sortidx]
      L = int(len(fnames) * ratio)
      if self._split in ['train']:
        split.append(sorted_image_ids[:L])
      else:
        split.append(sorted_image_ids[L:])
      print(lbl, sorted_image_ids)
      # sorted_tup = sorted(tup, key=lambda x: x[0])
      # print(sorted_tup)
      # if self._split in ['train']:
      #   split.extend([s[1] for s in sorted_tup[:L]])
      # else:
      #   split.extend([s[1] for s in sorted_tup[L:]])
    split = np.concatenate(split, axis=0)
    return split

  def get_cache_path(self):
    """Gets cache file name."""
    cache_path = os.path.join(self._folder, '_v2_cache.pkl')
    return cache_path

  def read_cache(self, cache_path):
    """Reads dataset from cached pkl file.

    Args:
      cache_path: filename of the cache.

    Returns:
      dict: data dictionary, None if the cache doesn't exist.
    """
    if os.path.exists(cache_path):
      try:
        with open(cache_path, 'rb') as f:
          data = pkl.load(f, encoding='bytes')
          images = data[b'images']
          labels = data[b'labels']
          label_str = data[b'label_str']
          fname_list = data[b'fname_list']
      except:  # NOQA
        with open(cache_path, 'rb') as f:
          data = pkl.load(f)
          images = data['images']
          labels = data['labels']
          label_str = data['label_str']
          fname_list = data['fname_list']
      return {
          'images': images,
          'labels': labels,
          'label_str': label_str,
          'fname_list': fname_list
      }
    else:
      return None

  def _read_dataset(self):
    label_idx = []
    label_str = []
    images = []
    fname_list = []
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
        fname_list.append(int(img_fname.split('.')[-2].split('_')[-1]))
    label_idx = np.array(label_idx)
    data = {
        'images': images,
        'labels': label_idx,
        'label_str': label_str,
        'fname_list': fname_list
    }
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
