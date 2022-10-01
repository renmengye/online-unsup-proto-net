# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data preprocessing and augmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
# from inspect import BoundArguments
# from absl import flags
import numpy as np

import tensorflow.compat.v1 as tf1  # NOQA
import tensorflow as tf
import tensorflow_addons as tfa

CROP_PROPORTION = 0.875  # Standard for ImageNet.


def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""
  return tf1.cond(
      tf1.less(
          tf1.random_uniform([], minval=0, maxval=1, dtype=tf1.float32),
          tf1.cast(p, tf1.float32)), lambda: func(x), lambda: x)


def random_apply_batch(func, p, x):
  """Randomly apply function func to x with probability p."""
  shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
  return tf.where(
      tf.less(
          tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32),
          tf.cast(p, tf.float32)), func(x), x)


# def random_apply_v2(func, p, x, x2):
#   """Randomly apply function func to x with probability p."""
#   return tf1.cond(
#       tf1.less(
#           tf1.random_uniform([], minval=0, maxval=1, dtype=tf1.float32),
#           tf1.cast(p, tf1.float32)), lambda: func(x), lambda: (x, x2))


def random_apply_v2(func, p, x, x2):
  """Randomly apply function func to x with probability p."""
  return tf1.cond(
      tf1.less(
          tf1.random_uniform([], minval=0, maxval=1, dtype=tf1.float32),
          tf1.cast(p, tf1.float32)), lambda: func(x, x2), lambda: x)


def random_brightness(image, max_delta, impl='simclrv2'):
  """A multiplicative vs additive change of brightness."""
  if impl == 'simclrv2':
    factor = tf1.random_uniform([], tf1.maximum(1.0 - max_delta, 0),
                                1.0 + max_delta)
    image = image * factor
  elif impl == 'simclrv1':
    image = random_brightness(image, max_delta=max_delta)
  else:
    raise ValueError('Unknown impl {} for random brightness.'.format(impl))
  return image


def to_grayscale(image, keep_channels=True):
  image = tf1.image.rgb_to_grayscale(image)
  if keep_channels:
    image = tf1.tile(image, [1] * (len(image.shape) - 1) + [3])
  return image


def color_jitter(image, strength, random_order=True):
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.

  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength
  if random_order:
    return color_jitter_rand(image, brightness, contrast, saturation, hue)
  else:
    return color_jitter_nonrand(image, brightness, contrast, saturation, hue)


def color_jitter_batch(image, strength, random_order=True):
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.

  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength
  if random_order:
    return color_jitter_rand_batch(image, brightness, contrast, saturation,
                                   hue)
  else:
    return color_jitter_nonrand_batch(image, brightness, contrast, saturation,
                                      hue)


def color_jitter_nonrand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is fixed).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf1.name_scope('distort_color'):

    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = random_brightness(x, max_delta=brightness)
      elif contrast != 0 and i == 1:
        x = tf1.image.random_contrast(
            x, lower=1 - contrast, upper=1 + contrast)
      elif saturation != 0 and i == 2:
        x = tf1.image.random_saturation(
            x, lower=1 - saturation, upper=1 + saturation)
      elif hue != 0:
        x = tf1.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf1.clip_by_value(image, 0., 1.)
    return image


def color_jitter_nonrand_batch(image,
                               brightness=0,
                               contrast=0,
                               saturation=0,
                               hue=0):
  """Distorts the color of the image (jittering order is fixed).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf1.name_scope('distort_color'):

    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = random_brightness(x, max_delta=brightness)
      elif contrast != 0 and i == 1:
        x = tf1.image.random_contrast(
            x, lower=1 - contrast, upper=1 + contrast)
      elif saturation != 0 and i == 2:
        x = tf1.image.random_saturation(
            x, lower=1 - saturation, upper=1 + saturation)
      elif hue != 0:
        x = tf1.image.random_hue(x, max_delta=hue)
      return x

    # images = []
    # images = tf.TensorArray(image.dtype, size=image.shape[0])

    def rnd_transform(x):
      for i in range(4):
        x = apply_transform(i, x, brightness, contrast, saturation, hue)
        x = tf1.clip_by_value(x, 0., 1.)
      return x

    # for j in range(image.shape[0]):
    #   image_ = image[j]
    #   for i in range(4):
    #     image_ = apply_transform(i, image_, brightness, contrast, saturation,
    #                              hue)
    #     image_ = tf1.clip_by_value(image_, 0., 1.)
    #   images = images.write(j, image_)
    # return images.stack()

    return tf.map_fn(rnd_transform, image)


def color_jitter_rand_batch(image,
                            brightness=0,
                            contrast=0,
                            saturation=0,
                            hue=0):
  """Distorts the color of the image (jittering order is random).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf1.name_scope('distort_color'):

    def apply_transform(i, x):
      """Apply the i-th transformation."""

      def brightness_foo():
        if brightness == 0:
          return x
        else:
          return random_brightness(x, max_delta=brightness)

      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf1.image.random_contrast(
              x, lower=1 - contrast, upper=1 + contrast)

      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf1.image.random_saturation(
              x, lower=1 - saturation, upper=1 + saturation)

      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf1.image.random_hue(x, max_delta=hue)

      x = tf1.cond(
          tf1.less(i, 2),
          lambda: tf1.cond(tf1.less(i, 1), brightness_foo, contrast_foo),
          lambda: tf1.cond(tf1.less(i, 3), saturation_foo, hue_foo))
      return x

    def rnd_transform(x):
      perm = tf1.random_shuffle(tf1.range(4))
      for i in range(4):
        x = apply_transform(perm[i], x)
        x = tf1.clip_by_value(x, 0., 1.)
      return x

    # images = tf.TensorArray(image.dtype, size=image.shape[0])
    # for j in range(image.shape[0]):
    #   image_ = image[j]
    #   perm = tf1.random_shuffle(tf1.range(4))
    #   for i in range(4):
    #     image_ = apply_transform(perm[i], image_)
    #     image_ = tf1.clip_by_value(image_, 0., 1.)
    #   images = images.write(j, image_)
    # return images.stack()
    return tf.map_fn(rnd_transform, image)


def color_jitter_rand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is random).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf1.name_scope('distort_color'):

    def apply_transform(i, x):
      """Apply the i-th transformation."""

      def brightness_foo():
        if brightness == 0:
          return x
        else:
          return random_brightness(x, max_delta=brightness)

      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf1.image.random_contrast(
              x, lower=1 - contrast, upper=1 + contrast)

      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf1.image.random_saturation(
              x, lower=1 - saturation, upper=1 + saturation)

      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf1.image.random_hue(x, max_delta=hue)

      x = tf1.cond(
          tf1.less(i, 2),
          lambda: tf1.cond(tf1.less(i, 1), brightness_foo, contrast_foo),
          lambda: tf1.cond(tf1.less(i, 3), saturation_foo, hue_foo))
      return x

    perm = tf1.random_shuffle(tf1.range(4))
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf1.clip_by_value(image, 0., 1.)
    return image


def _compute_crop_shape(image_height, image_width, aspect_ratio,
                        crop_proportion):
  """Compute aspect ratio-preserving shape for central crop.

  The resulting shape retains `crop_proportion` along one side and a proportion
  less than or equal to `crop_proportion` along the other side.

  Args:
    image_height: Height of image to be cropped.
    image_width: Width of image to be cropped.
    aspect_ratio: Desired aspect ratio (width / height) of output.
    crop_proportion: Proportion of image to retain along the less-cropped side.

  Returns:
    crop_height: Height of image after cropping.
    crop_width: Width of image after cropping.
  """
  image_width_float = tf1.cast(image_width, tf1.float32)
  image_height_float = tf1.cast(image_height, tf1.float32)

  def _requested_aspect_ratio_wider_than_image():
    crop_height = tf1.cast(
        tf1.rint(crop_proportion / aspect_ratio * image_width_float),
        tf1.int32)
    crop_width = tf1.cast(
        tf1.rint(crop_proportion * image_width_float), tf1.int32)
    return crop_height, crop_width

  def _image_wider_than_requested_aspect_ratio():
    crop_height = tf1.cast(
        tf1.rint(crop_proportion * image_height_float), tf1.int32)
    crop_width = tf1.cast(
        tf1.rint(crop_proportion * aspect_ratio * image_height_float),
        tf1.int32)
    return crop_height, crop_width

  return tf1.cond(aspect_ratio > image_width_float / image_height_float,
                  _requested_aspect_ratio_wider_than_image,
                  _image_wider_than_requested_aspect_ratio)


def center_crop(image, height, width, crop_proportion):
  """Crops to center of image and rescales to desired size.

  Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.

  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = tf1.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, height / width, crop_proportion)
  offset_height = ((image_height - crop_height) + 1) // 2
  offset_width = ((image_width - crop_width) + 1) // 2
  image = tf1.image.crop_to_bounding_box(image, offset_height, offset_width,
                                         crop_height, crop_width)
  image = tf1.image.resize_bicubic([image], [height, width])[0]
  bbox = tf1.constant([offset_height, offset_width, crop_height, crop_width],
                      dtype=tf1.float32)
  bbox /= tf1.constant([image_height, image_width, image_height, image_width],
                       dtype=tf1.float32)
  return image


def center_crop_batch(images, height, width, crop_proportion):
  """Crops to center of image and rescales to desired size.

  Args:
    image: Image Tensor to crop. [B, H, W, C]
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.

  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = tf1.shape(images)
  image_height = shape[1]
  image_width = shape[2]
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, height / width, crop_proportion)
  offset_height = ((image_height - crop_height) + 1) // 2
  offset_width = ((image_width - crop_width) + 1) // 2
  images = tf1.image.crop_to_bounding_box(images, offset_height, offset_width,
                                          crop_height, crop_width)
  images = tf1.image.resize_bicubic(images, [height, width])
  bbox = tf1.constant([offset_height, offset_width, crop_height, crop_width],
                      dtype=tf1.float32)
  bbox /= tf1.constant([image_height, image_width, image_height, image_width],
                       dtype=tf1.float32)
  return images


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf1.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: `Tensor` of image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf1.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    shape = tf1.shape(image)
    # print(aspect_ratio_range)
    # assert False
    sample_distorted_bounding_box = tf1.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf1.unstack(bbox_begin)
    target_height, target_width, _ = tf1.unstack(bbox_size)
    image = tf1.image.crop_to_bounding_box(image, offset_y, offset_x,
                                           target_height, target_width)
    bbox = tf1.stack([offset_y, offset_x, target_height, target_width])
    H = tf1.cast(shape[0], tf1.float32)
    W = tf1.cast(shape[1], tf1.float32)
    bbox = tf1.cast(bbox, tf1.float32) / tf1.stack([H, W, H, W])
    # tf.print('bbox', bbox)
    # tf.print('min_object_covered', min_object_covered)
    # tf.print('area range', area_range)
    return image


def sample_distorted_bounding_box_batch(shape,
                                        bbox=None,
                                        min_object_covered=0.1,
                                        aspect_ratio_range=(0.75, 1.33),
                                        area_range=(0.05, 1.0)):
  """Sample bound boxes.

  Args:
    shape: [B, H, W, C]
    bbox: [B, 4] ymin, xmin, ymax, xmax. [0, 1].
    aspect_ratio_range:
    area_range:
  """
  B = shape[0]
  H = tf.cast(shape[1], tf.float32)
  W = tf.cast(shape[2], tf.float32)
  aspect_ratio = tf.random.uniform(
      [B],
      minval=tf.cast(aspect_ratio_range[0], tf.float32),
      maxval=tf.cast(aspect_ratio_range[1], tf.float32),
      dtype=tf.float32)
  area = tf.random.uniform([B],
                           minval=tf.cast(area_range[0], tf.float32),
                           maxval=tf.cast(area_range[1], tf.float32),
                           dtype=tf.float32)
  wh_ratio = W / H  # []
  wh_ratio2 = aspect_ratio * wh_ratio  # [B]
  h = tf.math.sqrt(area * H * W / wh_ratio2)  # [B]
  w = h * wh_ratio2  # [B]
  minval = tf.zeros([B, 2])
  maxval_j = H - h  # [B]
  maxval_i = W - w  # [B]
  maxval = tf.stack([maxval_j, maxval_i], axis=-1)  # [B, 2]

  # Make sure that the box is included.
  if bbox is not None:
    box_w = (bbox[:, 3] - bbox[:, 1])
    box_h = (bbox[:, 2] - bbox[:, 0])
    box_area = box_w * box_h
    box_ratio = box_w / box_h
    box_h_min = tf.sqrt(min_object_covered * box_area / box_ratio)
    box_w_min = box_ratio * box_h_min
    max_start_j = (bbox[:, 2] - box_h_min) * H
    max_start_i = (bbox[:, 3] - box_w_min) * W
    max_box_start = tf.stack([max_start_j, max_start_i], axis=-1)  # [B, 2]
    min_end_j = (bbox[:, 0] + box_h_min) * H
    min_end_i = (bbox[:, 1] + box_w_min) * W
    min_start_j = min_end_j - h
    min_start_i = min_end_i - w
    min_box_start = tf.stack([min_start_j, min_start_i], axis=-1)  # [B, 2]
    minval = tf.maximum(min_box_start, minval)
    maxval = tf.minimum(max_box_start, maxval)

  begin = tf.random.uniform([B, 2], minval=minval, maxval=maxval)  # [B, 2]
  size = tf.stack([h, w], axis=-1)  # [B, 2]
  return begin, size


def distorted_bounding_box_crop_batch(images,
                                      bbox=None,
                                      min_object_covered=0.1,
                                      aspect_ratio_range=(0.75, 1.33),
                                      area_range=(0.05, 1.0),
                                      scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf1.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: `Tensor` of image data. [B, H, W, C]
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf1.name_scope(scope, 'distorted_bounding_box_crop', [images]):
    shape = tf.shape(images)
    bbox_begin, bbox_size = sample_distorted_bounding_box_batch(
        shape,
        bbox=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range)

    # Compute flow.
    B = shape[0]
    H = shape[1]
    W = shape[2]
    Hf = tf.cast(H, tf.float32)
    Wf = tf.cast(W, tf.float32)
    flow = tf.zeros([B, H, W, 2])
    flow_j = tf.cast(tf.range(H), tf.float32)
    flow_i = tf.cast(tf.range(W), tf.float32)

    flow_j = flow_j[None, :] / (Hf - 1) * (bbox_size[:, 0:1] - Hf)  # [B, H]
    flow_j += bbox_begin[:, 0:1]  # [B, H]
    flow_j = flow_j[:, :, None, None]  # [B,H,1,1]
    # print(flow_j.shape)
    # assert False
    flow_j = tf.tile(flow_j, [1, 1, W, 1])  # [B, H, W, 1]

    flow_i = flow_i[None, :] / (Wf - 1) * (bbox_size[:, 1:2] - Wf)  # [B, W]
    flow_i += bbox_begin[:, 1:2]  # [B, W]
    flow_i = flow_i[:, None, :, None]  # [B, 1, W, 1]
    flow_i = tf.tile(flow_i, [1, H, 1, 1])  # [B, H, W, 1]
    flow = tf.concat([flow_j, flow_i], axis=-1)  # [B, H, W, 2]

    images = tfa.image.dense_image_warp(images, -flow)
    bbox = tf.stack([bbox_begin, bbox_size], axis=-1)
    return images


def crop_and_resize(image,
                    bbox,
                    height,
                    width,
                    min_object_covered=0.1,
                    area_range=(0.08, 1.0)):
  """Make a random crop and resize it to height `height` and width `width`.

  Args:
    image: Tensor representing the image.
    height: Desired image height.
    width: Desired image width.

  Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
  """
  if bbox is None:
    bbox = tf1.constant([0.0, 0.0, 1.0, 1.0],
                        dtype=tf1.float32,
                        shape=[1, 1, 4])
  else:
    bbox = tf.reshape(bbox, [1, 1, 4])
  image = distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=(0.75, 1.33),
      area_range=area_range,
      max_attempts=100,
      scope=None)
  return tf1.image.resize_bicubic([image], [height, width])[0]


def crop_and_resize_batch(image,
                          bbox=None,
                          min_object_covered=0.1,
                          area_range=(0.08, 1.0)):
  """Make a random crop and resize it to height `height` and width `width`.

  Args:
    image: Tensor representing the image.

  Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
  """
  width = tf.shape(image)[2]
  height = tf.shape(image)[1]
  aspect_ratio = width / height
  images = distorted_bounding_box_crop_batch(
      image,
      bbox=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
      area_range=area_range,
      scope=None)
  return images


def get_motion_blur_kernel(kernel_size, angle, direction):
  """Add motion blurring to the given image with separable convolution."""
  """Code adapted from Kornia."""
  ks = tf.cast(kernel_size, tf.float32)
  if not isinstance(angle, tf.Tensor):
    angle = tf.constant(angle)
  if not isinstance(direction, tf.Tensor):
    direction = tf.constant(direction)

  angle = angle[None]
  direction = direction[None]
  d = (tf.clip_by_value(direction, -1.0, 1.0) + 1.0) / 2.0
  ks_i = tf.cast(tf.range(kernel_size), d.dtype)
  k = (d + ((1 - 2 * d) / (ks - 1)) * ks_i)
  kernel = tf.pad(k[:, None, None],
                  [[0, 0], [kernel_size // 2, kernel_size // 2], [0, 0]])
  kernel = tfa.image.rotate(kernel, angle / 180.0 * np.pi)
  kernel = kernel[:, :, None]
  kernel = kernel / tf.reduce_sum(kernel, [0, 1], keepdims=True)
  return kernel


def filter2d(input, kernel):
  """
  Args:
    input: [H, W, C]
    kernel: [kH, kW, 1, 1]
  """
  kernel = tf.tile(kernel, [1, 1, input.shape[-1], 1])
  return tf.nn.depthwise_conv2d(
      input[None], kernel, padding='SAME', strides=[1, 1, 1, 1])[0]


def random_motion_blur(image, height, width, p=1.0):
  del width

  def _transform(image):
    length = tf1.random.uniform([],
                                1,
                                tf.cast(height // 20, tf.int64),
                                dtype=tf1.int64)
    length = length * 2 + 1  # Odd number.
    angle = tf1.random.uniform([], 0, 360, dtype=tf1.float32)
    direction = tf.clip_by_value(
        tf1.random.normal([], 0.0, 0.5, dtype=tf1.float32), -1.0, 1.0)
    k = get_motion_blur_kernel(length, angle, direction)
    return filter2d(image, k)

  return random_apply(_transform, p=p, x=image)


def random_rotate(image, rotate_range):
  angle = tf.random.uniform([], minval=rotate_range[0], maxval=rotate_range[1])
  angle = angle / 180.0 * np.pi
  image = tfa.image.rotate(image, angle, 'BILINEAR')
  return image


def random_rotate_batch(images, rotate_range):
  angle = tf.random.uniform([tf.shape(images)[0]],
                            minval=rotate_range[0],
                            maxval=rotate_range[1])
  angle = angle / 180.0 * np.pi
  images = tfa.image.rotate(images, angle, 'BILINEAR')
  return images


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
  """Blurs the given image with separable convolution.


  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf1.to_int32(kernel_size / 2)
  kernel_size = radius * 2 + 1
  x = tf1.to_float(tf1.range(-radius, radius + 1))
  blur_filter = tf1.exp(
      -tf1.pow(x, 2.0) / (2.0 * tf1.pow(tf1.to_float(sigma), 2.0)))
  blur_filter /= tf1.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf1.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf1.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf1.shape(image)[-1]
  blur_h = tf1.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf1.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf1.expand_dims(image, axis=0)
  blurred = tf1.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf1.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf1.squeeze(blurred, axis=0)
  return blurred


def random_crop_with_resize(image,
                            bbox,
                            height,
                            width,
                            p=1.0,
                            min_object_covered=0.1,
                            area_range=(0.08, 1.0)):
  """Randomly crop and resize an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: Probability of applying this transformation.

  Returns:
    A preprocessed image `Tensor`.
  """

  # bbox = tf1.constant([0.0, 0.0, 1.0, 1.0])

  def _transform(image):  # pylint: disable=missing-docstring
    return crop_and_resize(
        image,
        bbox,
        height,
        width,
        min_object_covered=min_object_covered,
        area_range=area_range)

  return random_apply(_transform, p=p, x=image)


def random_crop_with_resize_batch(images,
                                  bbox=None,
                                  min_object_covered=0.1,
                                  p=1.0,
                                  area_range=(0.08, 1.0)):
  """Randomly crop and resize an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    p: Probability of applying this transformation.

  Returns:
    A preprocessed image `Tensor`.
  """

  def _transform(image):  # pylint: disable=missing-docstring
    result = crop_and_resize_batch(
        image,
        bbox=bbox,
        min_object_covered=min_object_covered,
        area_range=area_range)
    return result

  return _transform(images)

  # return random_apply(_transform, p=p, x=images)


def random_color_jitter(image, p=1.0, strength=1.0):

  def _transform(image):
    color_jitter_t = functools.partial(color_jitter, strength=strength)
    image = random_apply(color_jitter_t, p=0.8, x=image)
    return random_apply(to_grayscale, p=0.2, x=image)

  return random_apply(_transform, p=p, x=image)


def random_color_jitter_batch(image, p=1.0, strength=1.0):

  def _transform(image):
    color_jitter_t = functools.partial(color_jitter_batch, strength=strength)
    image = random_apply_batch(color_jitter_t, p=0.8, x=image)
    return random_apply_batch(to_grayscale, p=0.2, x=image)

  return random_apply_batch(_transform, p=p, x=image)


def random_blur(image, height, width, p=1.0):
  """Randomly blur an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: probability of applying this transformation.

  Returns:
    A preprocessed image `Tensor`.
  """
  del width

  def _transform(image):
    sigma = tf1.random.uniform([], 0.1, 1.2, dtype=tf1.float32)
    return gaussian_blur(
        image, kernel_size=height // 20, sigma=sigma, padding='SAME')

  return random_apply(_transform, p=p, x=image)


def batch_random_blur(images_list, height, width, blur_probability=0.5):
  """Apply efficient batch data transformations.

  Args:
    images_list: a list of image tensors.
    height: the height of image.
    width: the width of image.
    blur_probability: the probaility to apply the blur operator.

  Returns:
    Preprocessed feature list.
  """

  def generate_selector(p, bsz):
    shape = [bsz, 1, 1, 1]
    selector = tf1.cast(
        tf1.less(tf1.random_uniform(shape, 0, 1, dtype=tf1.float32), p),
        tf1.float32)
    return selector

  new_images_list = []
  # new_images_list = tf.TensorArray(tf.float32, size=len(images_list))
  for images in images_list:
    images_new = random_blur(images, height, width, p=1.)
    selector = generate_selector(blur_probability, tf1.shape(images)[0])
    images = images_new * selector + images * (1 - selector)
    images = tf1.clip_by_value(images, 0., 1.)
    new_images_list.append(images)

  return new_images_list


@tf.function
def preprocess_for_train(image,
                         bbox,
                         height,
                         width,
                         color_distort=True,
                         crop=True,
                         flip=True,
                         rotate=False,
                         rotate_degrees=10.0,
                         gaussian_blur=False,
                         gaussian_blur_prob=0.3,
                         motion_blur=False,
                         motion_blur_prob=0.3,
                         color_distort_strength=1.0,
                         min_object_covered=0.1,
                         area_range=(0.08, 1.0)):
  """Preprocesses the given image for training.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    color_distort: Whether to apply the color distortion.
    crop: Whether to crop the image.
    flip: Whether or not to flip left and right of an image.

  Returns:
    A preprocessed image `Tensor`.
  """
  nchannel = tf1.shape(image)[-1]
  if gaussian_blur:
    image = random_blur(
        image, tf.shape(image)[0], tf.shape(image)[1], p=gaussian_blur_prob)
  if motion_blur:
    image = random_motion_blur(
        image, tf.shape(image)[0], tf.shape(image)[1], p=motion_blur_prob)
  if rotate:
    image = random_rotate(
        image, rotate_range=[-rotate_degrees, rotate_degrees])
  if crop:
    image = random_crop_with_resize(
        image,
        bbox,
        height,
        width,
        min_object_covered=min_object_covered,
        area_range=area_range)
  if flip:
    image = tf1.image.random_flip_left_right(image)
  if color_distort:
    image_new = random_color_jitter(
        image[..., :3], strength=color_distort_strength)
    if tf.greater(nchannel, 3):
      image_mask = image[..., 3:]
      image = tf.concat([image_new, image_mask], axis=-1)
    else:
      image = image_new
  # image = tf1.reshape(image, [height, width, nchannel])
  image = tf1.clip_by_value(image, 0., 1.)
  return image


@tf.function
def preprocess_for_train_batch(image,
                               color_distort=True,
                               crop=True,
                               flip=True,
                               bbox=None,
                               rotate=False,
                               rotate_degrees=10.0,
                               gaussian_blur=False,
                               gaussian_blur_prob=0.2,
                               motion_blur=False,
                               motion_blur_prob=0.2,
                               min_object_covered=0.1,
                               color_distort_strength=1.0,
                               area_range=(0.08, 1.0)):
  """Preprocesses the given image for training.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    color_distort: Whether to apply the color distortion.
    crop: Whether to crop the image.
    flip: Whether or not to flip left and right of an image.

  Returns:
    A preprocessed image `Tensor`.
  """
  nchannel = tf1.shape(image)[-1]
  assert gaussian_blur is False, 'Not supported for batch'
  assert motion_blur is False, 'Not supported for batch'
  if rotate:
    image = random_rotate(
        image, rotate_range=[-rotate_degrees, rotate_degrees])
  if crop:
    image = random_crop_with_resize_batch(
        image,
        bbox=bbox,
        min_object_covered=min_object_covered,
        area_range=area_range)
  if flip:
    image = tf1.image.random_flip_left_right(image)

  if color_distort:
    image_new = random_color_jitter_batch(
        image[..., :3], strength=color_distort_strength)
    if tf.greater(nchannel, 3):
      image_mask = image[..., 3:]
      image = tf.concat([image_new, image_mask], axis=-1)
    else:
      image = image_new
  image = tf1.clip_by_value(image, 0., 1.)
  return image


# @tf.function
def preprocess_for_eval(image, height, width, crop=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    crop: Whether or not to (center) crop the test images.

  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
  image = tf1.reshape(image, [height, width, 3])
  image = tf1.clip_by_value(image, 0., 1.)
  return image


# @tf.function
def preprocess_for_eval_batch(image, height, width, crop=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    crop: Whether or not to (center) crop the test images.

  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = center_crop_batch(
        image, height, width, crop_proportion=CROP_PROPORTION)
  image = tf1.reshape(image, [height, width, 3])
  image = tf1.clip_by_value(image, 0., 1.)
  return image


@tf.function
def preprocess_image_batch2(image,
                            height,
                            width,
                            bbox=None,
                            min_object_covered=0.1,
                            is_training=False,
                            flip=True,
                            color_distort=True,
                            test_crop=True,
                            color_distort_strength=1.0,
                            area_range=(0.08, 1.0)):
  image = tf1.image.convert_image_dtype(image, dtype=tf1.float32)

  if is_training:
    x = preprocess_for_train_batch(
        image,
        bbox=bbox,
        min_object_covered=min_object_covered,
        color_distort=color_distort,
        flip=flip,
        color_distort_strength=color_distort_strength,
        area_range=area_range)
  else:
    x = preprocess_for_eval_batch(image, height, width, test_crop)

  return x


@tf.function
def preprocess_image_batch(image,
                           height,
                           width,
                           bbox=None,
                           min_object_covered=0.1,
                           is_training=False,
                           flip=True,
                           color_distort=True,
                           test_crop=True,
                           color_distort_strength=1.0,
                           area_range=(0.08, 1.0),
                           rotate=False,
                           rotate_degrees=10.0,
                           motion_blur=False,
                           motion_blur_prob=0.3,
                           gaussian_blur=False,
                           gaussian_blur_prob=0.3,
                           use_map_fn=True):

  def proc(i):
    return preprocess_image(
        image[i],
        height,
        width,
        bbox=bbox[i] if bbox is not None else None,
        min_object_covered=min_object_covered,
        is_training=is_training,
        flip=flip,
        color_distort=color_distort,
        test_crop=test_crop,
        color_distort_strength=color_distort_strength,
        area_range=area_range,
        rotate=rotate,
        rotate_degrees=rotate_degrees,
        motion_blur=motion_blur,
        motion_blur_prob=motion_blur_prob,
        gaussian_blur=gaussian_blur,
        gaussian_blur_prob=gaussian_blur_prob)

  if use_map_fn:
    return tf.map_fn(proc, tf.range(image.shape[0]), dtype=tf.float32)
  else:
    return preprocess_image_batch2(
        image,
        height,
        width,
        bbox=bbox,
        min_object_covered=min_object_covered,
        is_training=is_training,
        flip=flip,
        color_distort=color_distort,
        test_crop=test_crop,
        color_distort_strength=color_distort_strength,
        area_range=area_range,
        rotate=rotate,
        rotate_degrees=rotate_degrees)


@tf.function
def preprocess_image(image,
                     height,
                     width,
                     bbox=None,
                     min_object_covered=0.1,
                     is_training=False,
                     flip=True,
                     color_distort=True,
                     test_crop=True,
                     color_distort_strength=1.0,
                     area_range=(0.08, 1.0),
                     rotate=False,
                     rotate_degrees=10.0,
                     motion_blur=False,
                     motion_blur_prob=0.3,
                     gaussian_blur=False,
                     gaussian_blur_prob=0.3):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    is_training: `bool` for whether the preprocessing is for training.
    color_distort: whether to apply the color distortion.
    test_crop: whether or not to extract a central crop of the images
        (as for standard ImageNet evaluation) during the evaluation.

  Returns:
    A preprocessed image `Tensor` of range [0, 1].
  """
  image = tf1.image.convert_image_dtype(image, dtype=tf1.float32)

  if is_training:
    x = preprocess_for_train(
        image,
        bbox,
        height,
        width,
        color_distort=color_distort,
        flip=flip,
        color_distort_strength=color_distort_strength,
        min_object_covered=min_object_covered,
        area_range=area_range,
        rotate=rotate,
        rotate_degrees=rotate_degrees,
        motion_blur=motion_blur,
        motion_blur_prob=motion_blur_prob,
        gaussian_blur=gaussian_blur,
        gaussian_blur_prob=gaussian_blur_prob)
  else:
    x = preprocess_for_eval(image, height, width, test_crop)

  return x


if __name__ == '__main__':
  import cv2
  image = cv2.imread('./test.png').astype(np.float32) / 255.0
  print(image)
  assert image is not None
  image = tf.constant(image)
  for i in range(10):
    image2 = random_motion_blur(image, 100, 100, p=1.0)  # .numpy()
    image2 = random_blur(image2, 100, 100, p=1.0).numpy()
    image2 *= 255.0
    image2 = image2.astype(np.uint8)
    print(image2.shape)
    cv2.imwrite('./test-{}.png'.format(i), image2)
