""""Unit tests for blender."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest
import tensorflow as tf

from fewshot.data.preprocessors.simclr_utils import sample_distorted_bounding_box_batch  # NOQA
from fewshot.data.preprocessors.simclr_utils import distorted_bounding_box_crop_batch  # NOQA
from fewshot.data.preprocessors.simclr_utils import color_jitter  # NOQA


class SimclrUtilsTests(unittest.TestCase):

  def test_sample_distorted_bounding_box_batch(self):
    a = sample_distorted_bounding_box_batch([2, 28, 28, 1],
                                            aspect_ratio_range=(0.75, 1.33),
                                            area_range=(0.05, 1.0))
    print(a)

  def test_color_jitter(self):
    a = color_jitter(tf.zeros([2, 28, 28, 3]), 1.0)
    print(a.shape)

  def test_distorted_bounding_box_crop_batch(self):
    a = distorted_bounding_box_crop_batch(
        tf.zeros([2, 28, 28, 1]),
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0))
    print(a)


if __name__ == '__main__':
  unittest.main()
