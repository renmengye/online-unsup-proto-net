"""Iterators for Simulator episodes with large batches interleaving different
episodes together for self-supervised offline large batch learning.
In the end it will wrap into a single episode.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import os
import pickle as pkl

from fewshot.data.iterators.sim_episode_iterator import SimEpisodeIterator
from fewshot.data.preprocessors.simclr_preprocessor import SIMCLRPreprocessor
from fewshot.data.registry import RegisterIterator


def transform_bbox(bboxes, new_dim, original_dim=(600, 800)):
  bboxes = bboxes.reshape([-1, 2, 2]) * new_dim / np.array(original_dim)
  return bboxes.reshape([-1, 4]).astype(np.int64)


@RegisterIterator('matterport-category')
class MatterportCategoryIterator(SimEpisodeIterator):

  def __init__(self,
               dataset,
               config,
               batch_size,
               data_format="NCHW",
               rank=0,
               totalrank=1,
               **kwargs):
    self._config = config
    self._reset_flag = False
    self._data_format2 = data_format
    cls_dict_path = os.path.join(dataset.folder, "class_dict.pkl")
    if os.path.exists(cls_dict_path):
      class_dict = pkl.load(open(cls_dict_path, "rb"))
    else:
      raise ValueError("Class dict not found")
    self._class_dict = dict(zip(class_dict, range(len(class_dict))))
    super(MatterportCategoryIterator, self).__init__(
        dataset,
        config,
        None,
        batch_size,
        rank=rank,
        totalrank=totalrank,
        data_format="NHWC",
        **kwargs)
    print('my rank', rank, 'my total rank', totalrank)

  def _process_one(self, idx):
    """Process episodes.

    Args:
      idx: An integer ID.
    """
    raw = self.dataset.get_episode(idx)
    H = raw[0]['rgb'].shape[0]
    W = raw[0]['rgb'].shape[1]
    fcount = 0  # Frame counter
    # 4 channel image. The last channel is the object segmentation mask.
    # Allocate more than necessary first.
    N = sum([len(item['annotation'].keys()) for item in raw])
    x_s = np.zeros([N, H, W, 3], dtype=np.uint8)
    x_att = np.zeros([N, H, W, 1], dtype=np.uint8)
    # categories = np.zeros([N], dtype='|S15')
    categories = np.zeros([N], dtype=np.int64)
    instance_ids = np.zeros([N], dtype='|S15')
    bboxes = np.zeros([N, self._max_bbox_per_object, 4], dtype=np.int32)

    # Each frame has a centric object
    for item in raw:
      # Shuffle the objects in the frame.
      keys = list(item['annotation'].keys())
      if len(keys) > 0:
        for key in keys:

          # Skip black images.
          if np.sum(item['rgb']) == 0:
            continue

          x_s[fcount] = item['rgb']
          obj = item['annotation'][key]
          inst_id = obj['instance_id']
          annotation = item['annotation']
          x_s[fcount] = item['rgb']
          obj = annotation[key]
          inst_id = obj['instance_id']
          category = obj['category']
          instance_id = np.array(obj['instance_id'])
          bbox_i = np.array(obj['zoom_bboxes'])

          # Skip because it exeeds the total number of classes.
          attention_map = (item['instance_seg'] == inst_id).astype(np.uint8)

          if self._transform_bbox:
            bbox_i = transform_bbox(bbox_i, attention_map.shape[:2])

          if np.all(attention_map == 0):
            attention_bbox = bbox_i[-1]
            # print(attention_bbox)

            if np.all(attention_bbox == 0):
              # log.error("Bbox empty!")
              assert False

            attention_map = np.zeros_like(attention_map)
            y1, y2, x1, x2 = attention_bbox
            attention_map[y1:y2, x1:x2] = 1.0

          if len(np.nonzero(attention_map)[0]) == 0:
            continue

          x_att[fcount, :, :, 0] = attention_map * 255

          n_bboxes = bbox_i.shape[0]
          bboxes[fcount, :n_bboxes] = bbox_i

          categories[fcount] = self._class_dict[category]
          instance_ids[fcount] = instance_id
          fcount += 1

    x_s = x_s[:fcount]
    x_att = x_att[:fcount]
    categories = categories[:fcount]
    # print('categories', categories)

    if fcount == 0:
      return None

    episode = {
        'x_s': x_s,  # RGB image, mask as the 4th channel.
        'x_att': x_att,  # Attention mask.
        'y_s': categories  # Category
    }
    return episode

  def process_one(self, idx):
    """Process one episode.

    Args:
      Collection dictionary that contains the following keys:
        support: np.ndarray. Image ID in the support set.
        flag: np.ndarray. Binary flag indicating whether it is labeled (1) or
          unlabeled (0).
        query: np.ndarray. Image ID in the query set.
    """
    episode = self._process_one(idx)
    if episode is None:
      return None
    x_s = tf.image.convert_image_dtype(episode['x_s'], tf.float32)
    x_att = tf.image.convert_image_dtype(episode['x_att'], tf.float32)
    x = tf.concat([x_s, x_att], axis=-1)
    y = episode['y_s']
    return {'x': x, 'y': y, 'id': tf.zeros([x.shape[0]], dtype=tf.int32) + idx}

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    print('my rank 2', self._rank, 'my total rank', self._totalrank)
    dummy = self.process_one(0)
    dtype_dict = dict([(k, dummy[k].dtype) for k in dummy])
    shape_dict = dict(
        [(k, [None] + list(tf.shape(dummy[k])[1:])) for k in dummy])
    ds = tf.data.Dataset.from_generator(self.get_generator, dtype_dict,
                                        shape_dict)

    # Random shuffle queue.
    ds = ds.interleave(
        tf.data.Dataset.from_tensor_slices,
        cycle_length=20,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            buffer_size=4000)

    def preprocess(data):
      data['x'] = self.preprocessor(data['x'])
      return data

    ds = ds.batch(self.batch_size)
    if self.preprocessor is not None:
      ds = ds.map(preprocess)

    if self._prefetch:
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

  def get_generator(self):
    """Gets generator function, for tensorflow Dataset object."""
    while True:
      self._idx = np.arange(len(self.dataset))
      self._rnd.shuffle(self._idx)
      N = len(self._idx)
      nperrank = int(np.floor(N / self._totalrank))
      start = self._rank * nperrank
      end = (self._rank + 1) * nperrank
      self._idx = self._idx[start:end]
      self._reset_flag = False

      for i in self._idx:
        if not self._reset_flag:
          item = self.process_one(i)
          if item is not None:
            yield item
          else:
            break
        else:
          break

  def __len__(self):
    N = self.dataset.get_size()
    size = np.floor(N / self._totalrank) / float(self.batch_size)
    return int(np.floor(size))

  def reset(self):
    self._reset_flag = True

  @property
  def config(self):
    return self._config


if __name__ == '__main__':
  from fewshot.data.datasets.matterport import MatterportDataset
  from fewshot.data.samplers.minibatch_sampler import MinibatchSampler
  from fewshot.data.preprocessors.normalization_preprocessor import NormalizationPreprocessor  # NOQA
  from fewshot.experiments.utils import get_config
  from fewshot.configs.episode_config_pb2 import EpisodeConfig
  # from tqdm import tqdm
  sampler = MinibatchSampler(0)
  # class_dict = []
  # for key in ["train", "val", "test"]:
  #   dataset = MatterportDataset("./data/matterport3d/fewshot/h5_data", key)
  #   for idx in tqdm(range(len(dataset))):
  #     raw = dataset.get_episode(idx)
  #     for item in raw:
  #       keys = list(item['annotation'].keys())
  #       for key in keys:
  #         obj = item['annotation'][key]
  #         category = obj['category']
  #         if category not in class_dict:
  #           class_dict.append(category)
  #           print(class_dict)
  # class_dict = sorted(class_dict)
  # class_dict2 = dict(zip(range(len(class_dict)), class_dict))
  # print(class_dict2)
  # import pickle as pkl
  # pkl.dump(class_dict,
  #          open("./data/matterport3d/fewshot/h5_data/class_dict.pkl", "wb"))

  config = get_config(
      "./configs/episodes/roaming-rooms/roaming-rooms-100-siam-map.prototxt",
      EpisodeConfig)
  dataset = MatterportDataset("./data/matterport3d/fewshot/h5_data", "val")
  it = MatterportCategoryIterator(dataset, config, sampler, 30, prefetch=True)
  for d in it:
    print(d['y'])
    # for i in d['y']:
    #   print(i)
    #   print(str(i.numpy()))
    #   break
