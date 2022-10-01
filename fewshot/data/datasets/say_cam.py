"""SAYCam dataset API.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import os

import numpy as np
import cv2
import h5py

from fewshot.data.registry import RegisterDataset


@RegisterDataset("say-cam")
class SAYCamDataset(object):

  def __init__(self, folder, split):
    self._split = split
    self._folder = folder
    self._len_per_file = int(folder.split('_')[-1][:-1]) * 5
    all_h5_files = glob.glob(os.path.join(folder, split, "*.h5"))
    all_h5_files = sorted(all_h5_files)
    self._file_list = all_h5_files
    self._rnd = np.random.RandomState(0)

  def _make_iter(self, img_arr, l_arr, start=0, end=None, interval=1):
    """Makes an PNG encoding string iterator."""
    prev = 0
    if end is None:
      end = len(l_arr)
    l_cum = np.cumsum(l_arr)
    for i in range(start, end, interval):
      idx = l_cum[i]
      if i > 0:
        prev = l_cum[i - 1]
      else:
        prev = 0
      yield cv2.imdecode(img_arr[prev:idx], -1)

  def get_episode(self, idx, length, interval=1):
    """Get a single episode file."""
    M = self._len_per_file
    fidx = idx // M
    start = idx % M
    if fidx > len(self.file_list):
      return None
    filename = self.file_list[fidx]
    all_img = []
    with h5py.File(filename, "r") as f:
      end = start + length * interval
      selected = f["images"][start:end:interval]
      for i in selected:
        all_img.append(cv2.imdecode(i, -1))

    # Go find next episode.
    if len(all_img) < length:
      if fidx < len(self.file_list) - 1:
        filename2 = self.file_list[fidx + 1]
        with h5py.File(filename2, "r") as f:
          start = 0
          end = (length - len(all_img)) * interval
          selected = f["images"][start:end:interval]
          for i in selected:
            all_img.append(cv2.imdecode(i, -1))
      else:
        return None  # End of dataset.

    all_img = np.stack(all_img, axis=0)
    return {'x': all_img}

  def get_episode_old(self, idx, random_start=False, length=None, interval=1):
    """Get a single episode file."""
    filename = self.file_list[idx]
    all_img = []
    # print(idx, filename)
    with h5py.File(filename, "r") as f:
      num_img = len(f["images"])
      if length is None:
        start = 0
        end = num_img
      else:
        if random_start:
          xlen = num_img // interval
          assert xlen >= length, "Interval is too large"
          maxstart = xlen - length + 1
          start = self._rnd.randint(0, maxstart) * interval
          end = start + length * interval
        else:
          start = 0
          end = length * interval
      for i in f["images"][start:end:interval]:
        all_img.append(cv2.imdecode(i, -1))
      # print(start, end, random_start, num_img, interval, xlen, maxstart,
      #       length, len(all_img))
    all_img = np.stack(all_img, axis=0)
    return {'x': all_img}

  def get_episode_old2(self, idx, random_start=False, length=None, interval=1):
    """Get a single episode file."""
    filename = self.file_list[idx]
    all_img = []
    with h5py.File(filename, "r") as f:
      img = f["images"][:]
      img_len = f["images_len"][:]
      if length is None:
        start = 0
        end = len(img_len)
      else:
        if random_start:
          xlen = len(img_len) // interval
          maxstart = xlen - length + 1
          start = self._rnd.randint(0, maxstart) * interval
          end = start + length * interval
        else:
          start = 0
          end = length * interval
      img_iter = self._make_iter(
          img, img_len, start=start, end=end, interval=interval)
      for i in img_iter:
        all_img.append(i)
    all_img = np.stack(all_img, axis=0)
    return {'x': all_img}

  def get_size(self):
    """Get the total number of images."""
    if self.split == 'S':
      return 2959200  # TODO Estimated number, not the true number
    elif self.split == 'A':
      return 2112480
    elif self.split == 'Y':
      return 2109600
    else:
      raise ValueError("Unknown split")

  # def __len__(self):
  #   return len(self.file_list)

  def __len__(self):
    return len(self.file_list) * self._len_per_file

  @property
  def folder(self):
    """Data folder."""
    return self._folder

  @property
  def split(self):
    """Data split."""
    return self._split

  @property
  def file_list(self):
    return self._file_list


if __name__ == "__main__":
  for sp in ['S', 'A', 'Y']:
    dataset = SAYCamDataset("./data/say-cam/h5_data", sp)
    e = dataset.get_episode(0)
    print(e)
