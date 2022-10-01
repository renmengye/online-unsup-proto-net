"""
Training utilities.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import os
import sys
import tensorflow as tf
import time

from google.protobuf.text_format import Merge, MessageToString

from fewshot.data.data_factory import get_dataset


class ExperimentLogger():

  def __init__(self, writer):
    self._writer = writer

  def log(self, name, niter, value, family=None):
    tf.summary.scalar(name, float(value), step=niter)

  def flush(self):
    """Flushes results to disk."""
    self._writer.flush()

  def close(self):
    """Closes writer."""
    self._writer.close()


def save_config(config, save_folder, name='config.prototxt'):
  """Saves configuration to a file."""
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, name)
  with open(config_file, "w") as f:
    f.write(MessageToString(config))
  cmd_file = os.path.join(save_folder, "cmd-{}.txt".format(int(time.time())))
  if not os.path.exists(cmd_file):
    with open(cmd_file, "w") as f:
      f.write(' '.join(sys.argv))


def get_config(config_file, config_cls):
  """Reads configuration."""
  config = config_cls()
  Merge(open(config_file).read(), config)
  return config


def get_data(env_config, **kwargs):
  """Gets dataset."""
  result = {}
  global ALLIMAGE
  if env_config.dataset == 'celeb-a':
    kwargs['image_db'] = ALLIMAGE

  split_list = ['train', 'val', 'test']
  for split in split_list:
    if split == 'train':
      splitinfo = env_config.train_split
    elif split == 'val':
      splitinfo = env_config.val_split
    elif split == 'test':
      splitinfo = env_config.test_split

    if splitinfo is not None and len(splitinfo) > 0:
      print('split info', splitinfo)
      result[split] = get_dataset(env_config.dataset, env_config.data_folder,
                                  splitinfo, **kwargs)
    if env_config.dataset == 'celeb-a' and kwargs['image_db'] is None:
      kwargs['image_db'] = result[split]._images
      ALLIMAGE = result[split]._images
  result['metadata'] = env_config
  return result


def get_data_fs(env_config, load_train=False, load_val=True, load_test=True):
  """Gets few-shot dataset."""
  train_split = env_config.train_fs_split
  if train_split is None or (train_split == env_config.train_split and
                             not load_train):
    data_train_fs = None
  else:
    data_train_fs = get_dataset(env_config.dataset, env_config.data_folder,
                                env_config.train_fs_split)
  if env_config.val_fs_split is None or not load_val:
    data_val_fs = None
  else:
    data_val_fs = get_dataset(env_config.dataset, env_config.data_folder,
                              env_config.val_fs_split)
  if env_config.test_fs_split is None or not load_test:
    data_test_fs = None
  else:
    data_test_fs = get_dataset(env_config.dataset, env_config.data_folder,
                               env_config.test_fs_split)
  return {
      'train_fs': data_train_fs,
      'val_fs': data_val_fs,
      'test_fs': data_test_fs,
      'metadata': env_config
  }


def latest_file(folder, prefix):
  """Query the most recent checkpoint."""
  print('Searching latest file', folder, prefix)
  list_of_files = glob.glob(os.path.join(folder, prefix + '*'))
  if len(list_of_files) == 0:
    return None
  latest_file = max(
      list_of_files, key=lambda f: int(f.split('-')[-1].split('.')[0]))
  return latest_file


def delete_old_ckpt(folder, prefix, cur_step, max_history):
  fnames = glob.glob(os.path.join(folder, prefix + '*'))
  for f in fnames:
    name = f.split('-')[-1].split('.')[0]
    if cur_step - int(name) > max_history:
      os.remove(f)
      # print('deleted old {}'.format(f))
