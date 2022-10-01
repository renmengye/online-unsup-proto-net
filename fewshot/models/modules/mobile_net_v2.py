from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.backbone import Backbone
from fewshot.models.registry import RegisterModule
from fewshot.models.modules.keras_mobile_net_v2 import MobileNetV2Horovod
from fewshot.utils.logger import get as get_logger
log = get_logger()


@RegisterModule('mobilenet_v2')
class MobileNetV2Backbone(Backbone):

  def __init__(self, config, dtype=tf.float32):
    super(MobileNetV2Backbone, self).__init__(config, dtype=dtype)
    from tensorflow.keras import backend
    if config.data_format == "NCHW":
      backend.set_image_data_format("channels_first")
    elif config.data_format == "NHWC":
      backend.set_image_data_format("channels_last")
    # assert config.data_format == "NHWC"
    # self._inner_module = tf.keras.applications.mobilenet_v2.MobileNetV2(
    #     include_top=False, weights=None, pooling='avg')
    sync_batch_norm = config.normalization == "sync_batch_norm"
    assert sync_batch_norm
    self._inner_module = MobileNetV2Horovod(
        include_top=False,
        weights=None,
        pooling='avg',
        sync_batch_norm=sync_batch_norm,
        add_last_relu=config.add_last_relu)

  def forward(self, x, is_training=True, **kwargs):
    x = self._inner_module(x, training=is_training)
    return x

  def weights(self):
    return self._inner_module.weights

  def set_trainable(self, trainable):
    # self._inner_module.trainable = trainable
    # log.info('Set keras mobile net trainable={}'.format(trainable))
    for w in self.weights():
      w._trainable = trainable
      # log.info('{} trainable={}'.format(w.name, w.trainable))


if __name__ == '__main__':
  backbone = MobileNetV2Backbone(None)
  y = backbone(tf.ones([1, 224, 224, 3]))
  print(y)
