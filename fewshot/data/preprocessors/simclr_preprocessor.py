"""Data augmentation preprocessor for SIMCLR.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.data.preprocessors.preprocessor import Preprocessor
from fewshot.data.preprocessors.simclr_utils import preprocess_image
from fewshot.data.preprocessors.simclr_utils import preprocess_image_batch


class SIMCLRPreprocessor(Preprocessor):

  def __init__(self,
               crop_size_h,
               crop_size_w,
               color_distort=True,
               flip=True,
               color_distort_strength=1.0,
               area_range_lb=0.08,
               rotate=False,
               rotate_degrees=0,
               gaussian_blur=False,
               gaussian_blur_prob=0.0,
               motion_blur=False,
               motion_blur_prob=0.0,
               num_views=2,
               min_object_covered=0.1,
               map_fn=True):
    self._crop_size_h = crop_size_h
    self._crop_size_w = crop_size_w
    self._color_distort = color_distort
    self._color_distort_strength = color_distort_strength
    self._rotate = rotate
    self._rotate_degrees = rotate_degrees
    self._motion_blur = motion_blur
    self._motion_blur_prob = motion_blur_prob
    self._gaussian_blur = gaussian_blur
    self._gaussian_blur_prob = gaussian_blur_prob
    self._num_views = num_views
    self._area_range_lb = area_range_lb
    self._flip = flip
    self._min_object_covered = min_object_covered
    self._MAP_FN = map_fn

    assert num_views > 0

  def preprocess_batch(self, image, bbox=None):
    crop_size_h = self._crop_size_h
    crop_size_w = self._crop_size_w
    if self._num_views == 2:
      image = tf.concat([image, image], axis=0)
      if bbox is not None:
        bbox = tf.concat([bbox, bbox], axis=0)
      image = preprocess_image_batch(
          image,
          bbox=bbox,
          min_object_covered=self._min_object_covered,
          height=crop_size_h,
          width=crop_size_w,
          color_distort=self._color_distort,
          flip=self._flip,
          color_distort_strength=self._color_distort_strength,
          area_range=(self._area_range_lb, 1.0),
          rotate=self._rotate,
          rotate_degrees=self._rotate_degrees,
          motion_blur=self._motion_blur,
          motion_blur_prob=self._motion_blur_prob,
          gaussian_blur=self._gaussian_blur,
          gaussian_blur_prob=self._gaussian_blur_prob,
          is_training=True,
          use_map_fn=self._MAP_FN)
      return tf.concat(tf.split(image, 2, axis=0), axis=-1)
    elif self._num_views == 1:
      x = preprocess_image_batch(
          image,
          bbox=bbox,
          min_object_covered=self._min_object_covered,
          height=crop_size_h,
          width=crop_size_w,
          color_distort=self._color_distort,
          flip=self._flip,
          color_distort_strength=self._color_distort_strength,
          area_range=(self._area_range_lb, 1.0),
          rotate=self._rotate,
          rotate_degrees=self._rotate_degrees,
          motion_blur=self._motion_blur,
          motion_blur_prob=self._motion_blur_prob,
          gaussian_blur=self._gaussian_blur,
          gaussian_blur_prob=self._gaussian_blur_prob,
          is_training=True,
          use_map_fn=self._MAP_FN)
      return x
    else:
      image = tf.tile(image, [self._num_views, 1, 1, 1])
      if bbox is not None:
        bbox = tf.tile(bbox, [self._num_views, 1])
      image = preprocess_image_batch(
          image,
          bbox=bbox,
          min_object_covered=self._min_object_covered,
          height=crop_size_h,
          width=crop_size_w,
          color_distort=self._color_distort,
          flip=self._flip,
          color_distort_strength=self._color_distort_strength,
          area_range=(self._area_range_lb, 1.0),
          rotate=self._rotate,
          rotate_degrees=self._rotate_degrees,
          motion_blur=self._motion_blur,
          motion_blur_prob=self._motion_blur_prob,
          gaussian_blur=self._gaussian_blur,
          gaussian_blur_prob=self._gaussian_blur_prob,
          is_training=True,
          use_map_fn=self._MAP_FN)
      return tf.concat(tf.split(image, self._num_views, axis=0), axis=-1)

  def preprocess(self, image, bbox=None):
    crop_size_h = self._crop_size_h
    crop_size_w = self._crop_size_w
    if self._num_views == 2:
      x1 = preprocess_image(
          image,
          bbox=bbox,
          height=crop_size_h,
          width=crop_size_w,
          color_distort=self._color_distort,
          min_object_covered=self._min_object_covered,
          flip=self._flip,
          color_distort_strength=self._color_distort_strength,
          area_range=(self._area_range_lb, 1.0),
          rotate=self._rotate,
          rotate_degrees=self._rotate_degrees,
          motion_blur=self._motion_blur,
          motion_blur_prob=self._motion_blur_prob,
          gaussian_blur=self._gaussian_blur,
          gaussian_blur_prob=self._gaussian_blur_prob,
          is_training=True)
      x2 = preprocess_image(
          image,
          bbox=bbox,
          height=crop_size_h,
          width=crop_size_w,
          color_distort=self._color_distort,
          min_object_covered=self._min_object_covered,
          flip=self._flip,
          color_distort_strength=self._color_distort_strength,
          area_range=(self._area_range_lb, 1.0),
          rotate=self._rotate,
          rotate_degrees=self._rotate_degrees,
          motion_blur=self._motion_blur,
          motion_blur_prob=self._motion_blur_prob,
          gaussian_blur=self._gaussian_blur,
          gaussian_blur_prob=self._gaussian_blur_prob,
          is_training=True)
      # tf.print('views 1', tf.shape(x1))
      # tf.print('views 2', tf.shape(x2))
      # tf.print('views 3', tf.concat([x1, x2], axis=-1))
      return tf.concat([x1, x2], axis=-1)
    elif self._num_views == 1:
      x = preprocess_image(
          image,
          bbox=bbox,
          height=crop_size_h,
          width=crop_size_w,
          color_distort=self._color_distort,
          min_object_covered=self._min_object_covered,
          flip=self._flip,
          color_distort_strength=self._color_distort_strength,
          area_range=(self._area_range_lb, 1.0),
          rotate=self._rotate,
          rotate_degrees=self._rotate_degrees,
          motion_blur=self._motion_blur,
          motion_blur_prob=self._motion_blur_prob,
          gaussian_blur=self._gaussian_blur,
          gaussian_blur_prob=self._gaussian_blur_prob,
          is_training=True)
      return x
    else:
      views = []
      bbox = []
      for n in range(self._num_views):
        x = preprocess_image(
            image,
            bbox=bbox,
            height=crop_size_h,
            width=crop_size_w,
            color_distort=self._color_distort,
            min_object_covered=self._min_object_covered,
            flip=self._flip,
            color_distort_strength=self._color_distort_strength,
            area_range=(self._area_range_lb, 1.0),
            rotate=self._rotate,
            rotate_degrees=self._rotate_degrees,
            motion_blur=self._motion_blur,
            motion_blur_prob=self._motion_blur_prob,
            gaussian_blur=self._gaussian_blur,
            gaussian_blur_prob=self._gaussian_blur_prob,
            is_training=True)
        views.append(x)
        # tf.print('views', n, tf.shape(x))
      # tf.print('views', len(views))
      return tf.concat(views, axis=-1)
