from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import scipy.io
import os
import numpy as np

from fewshot.data.registry import RegisterDataset
from fewshot.data.datasets.dataset import Dataset


@RegisterDataset('omniglot-iid')
class OmniglotIIDDataset(Dataset):

  def __init__(self, folder, split):
    """Creates an omniglot dataset instance."""
    self._folder = folder
    self._split = split
    data = self.read_dataset()
    self._images = (data['images'] * 255.0).astype(np.uint8)
    self._labels = data['labels']

  def read_dataset(self):
    """Read data from folder or cache."""
    data = scipy.io.loadmat(os.path.join(self.folder, 'chardata.mat'))
    if self.split == 'train':
      return {
          'images': data['data'].T.reshape([-1, 28, 28, 1]),
          'labels': np.argmax(data['target'].T, axis=1)
      }
    elif self.split in ['val', 'test']:
      return {
          'images': data['testdata'].T.reshape([-1, 28, 28, 1]),
          'labels': np.argmax(data['testtarget'].T, axis=1)
      }
    return data

  def __len__(self):
    return len(self.images)

  def get_size(self):
    """Gets the total number of images."""
    return len(self.images)

  def get_ids(self):
    return np.arange(len(self.images))

  def get_images(self, inds):
    """Gets an image or a batch of images."""
    return self.images[inds]

  def get_labels(self, inds):
    """Gets the label of an image or a batch of images."""
    return self.labels[inds]

  @property
  def folder(self):
    """Data folder."""
    return self._folder

  @property
  def split(self):
    """Data split."""
    return self._split

  @property
  def images(self):
    """Image data."""
    return self._images

  @property
  def labels(self):
    """Label data."""
    return self._labels
