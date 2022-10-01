"""Iterators for few-shot episode.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.data.iterators.sim_episode_iterator import SimEpisodeIterator
from fewshot.data.preprocessors.simclr_preprocessor import SIMCLRPreprocessor
from fewshot.data.registry import RegisterIterator


def transform_bbox(bboxes, new_dim, original_dim=(600, 800)):
  bboxes = bboxes.reshape([-1, 2, 2]) * new_dim / np.array(original_dim)
  return bboxes.reshape([-1, 4]).astype(np.int64)


@RegisterIterator('mix-unsupervised-augment-sim-noniid')
class MixUnsupAugmentSimIteratorNonIID(SimEpisodeIterator):

  def __init__(self,
               dataset,
               config,
               batch_size,
               preprocessor=None,
               rank=0,
               totalrank=1,
               seed=0):
    self._config = config
    # self._rank = rank
    # self._totalrank = totalrank
    self._reset_flag = False
    self._maxlen = config.maxlen
    super(MixUnsupAugmentSimIteratorNonIID, self).__init__(
        dataset,
        config,
        None,
        batch_size,
        preprocessor=preprocessor,
        prefetch=False,
        random_crop=True,
        random_shuffle_objects=False,
        random_drop=False,
        random_flip=False,
        random_jitter=False,
        rank=rank,
        totalrank=totalrank,
        seed=seed)

  def _process_one(self, idx):
    """Process episodes.

    Args:
      idx: An integer ID.
    """
    raw = self.dataset.get_episode(idx)
    H = raw[0]['rgb'].shape[0]
    W = raw[0]['rgb'].shape[1]
    T = self.maxlen
    fcount = 0  # Frame counter
    # 4 channel image. The last channel is the object segmentation mask.
    # Allocate more than necessary first.
    N = sum([len(item['annotation'].keys()) for item in raw])
    x_s = np.zeros([N, H, W, 3], dtype=np.uint8)
    x_att = np.zeros([N, H, W, 1], dtype=np.uint8)
    categories = np.zeros([N], dtype='|S15')
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

          category = np.array(obj['category'])
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

          categories[fcount] = category
          instance_ids[fcount] = instance_id
          fcount += 1

    x_s = x_s[:fcount]
    x_att = x_att[:fcount]
    # Crop into T frames.
    if fcount >= T:
      # random sample T frames.
      if self._random_crop:
        start = self._rnd.randint(0, fcount - T + 1)
      else:
        start = 0
    else:
      return None
    x_s = x_s[start:start + T]
    x_att = x_att[start:start + T]

    # Cut the GPU specific chunk.
    if self._totalrank > 1:
      M = self._totalrank
      m = T // M
      start2 = self._rank * m
      end2 = (self._rank + 1) * m
      x_s = x_s[start2:end2]
      x_att = x_att[start2:end2]
      # print(self._rank, T, start2, end2)
    # else:
    #   start2 = 0
    #   end2 = T
    # print('Rank: {} ID: {} Count: {} Start: {} Start2: {} End2: {}'.format(
    #     self._rank, idx, fcount, start, start2, end2))

    if fcount == 0:
      return None

    episode = {
        'x_s': x_s,  # RGB image, mask as the 4th channel.
        'x_att': x_att  # Attention mask.
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
    # print(self._rank, idx)
    episode = self._process_one(idx)
    if episode is None:
      return None
    x_s = episode['x_s']
    x_att = episode['x_att']
    img_s = tf.concat([x_s, x_att], axis=-1)
    N = img_s.shape[0]
    H = img_s.shape[1]
    W = img_s.shape[2]
    C = img_s.shape[3]
    img_s = tf.image.convert_image_dtype(img_s, tf.float32)

    B = x_att.shape[0]
    bbox = np.zeros([B, 4], dtype=np.float32)

    for i in range(B):
      nonzero = np.nonzero(x_att[i])
      if len(nonzero[0]) > 0:
        bbox_ymin = float(nonzero[0].min()) / float(H)
        bbox_ymax = float(nonzero[0].max()) / float(H)
        bbox_xmin = float(nonzero[1].min()) / float(W)
        bbox_xmax = float(nonzero[1].max()) / float(W)
        bbox[i, 0] = bbox_ymin
        bbox[i, 1] = bbox_xmin
        bbox[i, 2] = bbox_ymax
        bbox[i, 3] = bbox_xmax
      else:
        bbox[i, 0] = 0.0
        bbox[i, 1] = 0.0
        bbox[i, 2] = 1.0
        bbox[i, 3] = 1.0

    return {
        'x': img_s,
        'bbox': bbox,
        'id': tf.zeros([img_s.shape[0]], dtype=tf.int32) + idx
    }

  def get_dataset(self):
    """Gets TensorFlow Dataset object."""
    dummy = self.process_one(0)
    dtype_dict = dict([(k, dummy[k].dtype) for k in dummy])
    shape_dict = dict(
        [(k, [None] + list(tf.shape(dummy[k])[1:])) for k in dummy])
    ds = tf.data.Dataset.from_generator(self.get_generator, dtype_dict,
                                        shape_dict)
    ds = ds.interleave(tf.data.Dataset.from_tensor_slices, cycle_length=1)

    img = dummy['x']
    crop_size_h = img.shape[1]
    crop_size_w = img.shape[2]
    simclr_preprocessor = SIMCLRPreprocessor(
        crop_size_h,
        crop_size_w,
        color_distort=self.config.color_distort,
        flip=self.config.flip_left_right,
        color_distort_strength=self.config.color_distort_strength,
        area_range_lb=self.config.area_lb,
        min_object_covered=self.config.min_object_covered,
        num_views=self.config.num_aug)

    def simclr_preprocess(data):
      x = simclr_preprocessor(data['x'], bbox=data['bbox'])
      return {'x': x, 'id': data['id']}

    def preprocess(data):
      data['x'] = self.preprocessor(data['x'])
      return data

    ds = ds.map(simclr_preprocess, deterministic=True)
    ds = ds.batch(self.batch_size)
    ds = ds.map(preprocess, deterministic=True)
    return ds

  def get_generator(self):
    """Gets generator function, for tensorflow Dataset object."""
    while True:
      self._idx = np.arange(len(self.dataset))
      self._rnd.shuffle(self._idx)
      # N = len(self._idx)
      # nperrank = int(np.floor(N / self._totalrank))
      # start = self._rank * nperrank
      # end = (self._rank + 1) * nperrank
      # self._idx = self._idx[start:end]
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
