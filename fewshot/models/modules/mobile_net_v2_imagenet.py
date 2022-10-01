from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.backbone import Backbone
from fewshot.models.registry import RegisterModule


@RegisterModule('mobilenet_v2_imagenet')
class MobileNetV2ImageNetBackbone(Backbone):

  def __init__(self, config, dtype=tf.float32):
    super(MobileNetV2ImageNetBackbone, self).__init__(config, dtype=dtype)
    # assert config.data_format == "NHWC"
    self._inner_module = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False, weights='imagenet', pooling='avg')

  def forward(self, x, is_training=True, **kwargs):
    x = self._inner_module(x, training=is_training)
    return x

  def weights(self):
    return self._inner_module.weights


if __name__ == '__main__':
  backbone = MobileNetV2ImageNetBackbone(None)
  y = backbone(tf.ones([1, 224, 224, 3]))
  print(y)
