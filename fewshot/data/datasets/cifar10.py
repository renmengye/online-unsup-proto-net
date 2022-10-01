from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import pickle as pkl
from six.moves import xrange
from fewshot.utils import logger
from fewshot.data.registry import RegisterDataset

log = logger.get()

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CLASSES = 10
NUM_CHANNEL = 3
NUM_TRAIN_IMG = 50000
NUM_TEST_IMG = 10000


def unpickle(file):
  fo = open(file, 'rb')
  dict = pkl.load(fo, encoding='bytes')
  fo.close()
  return dict


def read_CIFAR10(data_folder):
  """ Reads and parses examples from CIFAR10 data files """

  train_img = []
  train_label = []
  test_img = []
  test_label = []

  train_file_list = [
      "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
      "data_batch_5"
  ]
  test_file_list = ["test_batch"]

  for i in xrange(len(train_file_list)):
    tmp_dict = unpickle(
        os.path.join(data_folder, 'cifar-10-batches-py', train_file_list[i]))
    train_img.append(tmp_dict[b"data"])
    train_label.append(tmp_dict[b"labels"])

  tmp_dict = unpickle(
      os.path.join(data_folder, 'cifar-10-batches-py', test_file_list[0]))
  test_img.append(tmp_dict[b"data"])
  test_label.append(tmp_dict[b"labels"])

  train_img = np.concatenate(train_img)
  train_label = np.concatenate(train_label)
  test_img = np.concatenate(test_img)
  test_label = np.concatenate(test_label)

  train_img = np.reshape(
      train_img, [NUM_TRAIN_IMG, NUM_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH])
  test_img = np.reshape(test_img,
                        [NUM_TEST_IMG, NUM_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH])

  # change format from [B, C, H, W] to [B, H, W, C] for feeding to Tensorflow
  train_img = np.transpose(train_img, [0, 2, 3, 1])
  test_img = np.transpose(test_img, [0, 2, 3, 1])
  CIFAR10_data = {}
  CIFAR10_data["train_img"] = train_img
  CIFAR10_data["test_img"] = test_img
  CIFAR10_data["train_label"] = train_label
  CIFAR10_data["test_label"] = test_label
  return CIFAR10_data


@RegisterDataset('cifar-10')
class CIFAR10Dataset():

  def __init__(self, folder, split, num_fold=10, fold_id=0):
    self.split = split
    self.data = read_CIFAR10(folder)
    num_ex = 50000
    split_idx = np.arange(num_ex)
    rnd = np.random.RandomState(0)
    rnd.shuffle(split_idx)
    num_valid = int(np.ceil(num_ex / num_fold))
    valid_start = fold_id * num_valid
    valid_end = min((fold_id + 1) * num_valid, num_ex)
    valid_split_idx = split_idx[valid_start:valid_end]
    train_split_idx = np.concatenate(
        [split_idx[:valid_start], split_idx[valid_end:]])
    if split == 'val':
      self.idx = valid_split_idx
      self.images = self.data['train_img']
      self.labels = self.data['train_label']
    elif split == 'test':
      self.idx = np.arange(NUM_TEST_IMG)
      self.images = self.data['test_img']
      self.labels = self.data['test_label']
    elif split == 'train':
      self.idx = train_split_idx
      self.images = self.data['train_img']
      self.labels = self.data['train_label']
    elif split == 'trainval':
      self.idx = split_idx
      self.images = self.data['train_img']
      self.labels = self.data['train_label']

    # print(self.images.dtype, self.images.shape)
    # print(self.labels[:10], self.labels.dtype, self.labels.shape)
    # assert False

  def get_size(self):
    return len(self.idx)

  def get_ids(self):
    return self.idx

  def get_images(self, inds):
    """Gets an image or a batch of images."""
    return self.images[inds]

  def get_labels(self, inds):
    """Gets the label of an image or a batch of images."""
    return self.labels[inds]
