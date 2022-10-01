from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import glob
from fewshot.data.datasets.dataset import Dataset
from fewshot.data.registry import RegisterDataset


@RegisterDataset("say-cam-time")
class SAYCamTimeDataset(Dataset):

  def __init__(self, folder, split):
    super().__init__()
    self.folder = folder
    self.split = split

  def get_filenames(self):
    filenames = glob.glob(os.path.join(self.folder, '*.h5'))
    N = len(filenames)
    if self.split == 'train':
      filenames = filenames[:N // 2]
    else:
      filenames = filenames[N // 2:]
    return filenames

  def __len__(self):
    return len(self.get_filenames()) * 1440

  def get_images(self, inds):
    """Gets images based on indices."""
    raise NotImplementedError()

  def get_labels(self, inds):
    """Gets labels based on indices."""
    raise NotImplementedError()

  def get_size(self):
    """Gets the size of the dataset."""
    return len(self.get_filenames()) * 1440
