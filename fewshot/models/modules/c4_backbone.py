"""Basic 4-layer convolution network backbone.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.backbone import Backbone
from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.nnlib import Conv2D, BatchNorm
from fewshot.models.modules.group_norm import GroupNorm
from fewshot.models.modules.sync_batch_norm import SyncBatchNorm
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope
from fewshot.utils.logger import get as get_logger

log = get_logger()


class ConvModule(ContainerModule):

  def __init__(self,
               name,
               in_filter,
               out_filter,
               stride=2,
               add_relu=True,
               data_format="NCHW",
               pool_padding="SAME",
               normalization="batch_norm",
               num_groups=8,
               dtype=tf.float32,
               wdict=None):
    super(ConvModule, self).__init__()
    self._data_format = data_format
    self._name = name
    with variable_scope(name):
      self._conv = Conv2D(
          "conv",
          3,
          in_filter,
          out_filter,
          self._stride_arr(1),
          data_format=data_format,
          dtype=dtype,
          wdict=wdict)
      if normalization == "batch_norm":
        self._norm = BatchNorm(
            "bn",
            out_filter,
            data_format=data_format,
            dtype=dtype,
            wdict=wdict)
      elif normalization == "group_norm":
        self._norm = GroupNorm(
            "gn",
            out_filter,
            num_groups,
            data_format=data_format,
            dtype=dtype,
            wdict=wdict)
      elif normalization == "sync_batch_norm":
        self._norm = SyncBatchNorm(
            "bn",
            out_filter,
            data_format=data_format,
            dtype=dtype,
            wdict=wdict)
      else:
        raise ValueError("Unknown normalization: {}".format(normalization))
    self._stride = stride
    self._add_relu = add_relu
    self._pool_padding = pool_padding

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    if self._data_format == 'NCHW':
      return [1, 1, stride, stride]
    else:
      return [1, stride, stride, 1]

  def forward(self, x, is_training=tf.constant(True), **kwargs):
    x = self._conv(x)
    x = self._norm(x, is_training=is_training)
    if self._add_relu:
      x = tf.nn.relu(x)
    if self._stride > 1:
      x = tf.nn.max_pool(
          x,
          self._stride_arr(self._stride),
          self._stride_arr(self._stride),
          padding=self._pool_padding,
          data_format=self._data_format)
      # if self._add_relu:
      #   x = tf.nn.max_pool(
      #       x,
      #       self._stride_arr(self._stride),
      #       self._stride_arr(self._stride),
      #       padding=self._pool_padding,
      #       data_format=self._data_format)
      # else:
      #   x = tf.nn.avg_pool(
      #       x,
      #       self._stride_arr(self._stride),
      #       self._stride_arr(self._stride),
      #       padding=self._pool_padding,
      #       data_format=self._data_format)
    # tf.summary.histogram(self._name, x)
    return x


@RegisterModule("c4_backbone")
class C4Backbone(Backbone):

  def __init__(self, config, wdict=None):
    super(C4Backbone, self).__init__(config)
    self._config = config
    assert len(config.pool) == 0
    # assert config.add_last_relu
    L = len(config.num_filters)
    if len(config.pool) > 0:
      pool = config.pool
    else:
      pool = [2] * L
    if config.normalization == "group_norm":
      num_groups = config.num_groups
    else:
      num_groups = [8] * L
    in_filters = [config.num_channels] + config.num_filters[:-1]
    add_relu = [True] * (L - 1) + [config.add_last_relu]
    self._conv_list = []

    for i in range(len(config.num_filters)):
      self._conv_list.append(
          ConvModule(
              "conv{}".format(i + 1),
              in_filters[i],
              config.num_filters[i],
              stride=pool[i],
              add_relu=add_relu[i],
              data_format=config.data_format,
              pool_padding=config.pool_padding,
              normalization=config.normalization,
              num_groups=num_groups[i],
              wdict=wdict))

  def forward(self, x, is_training, **kwargs):
    # tf.print(tf.reduce_max(x), tf.reduce_min(x))
    for m in self._conv_list:
      x = m(x, is_training=is_training)
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    if self.config.activation_scaling > 0:
      x = x * self.config.activation_scaling

    if self.config.add_dropout and is_training:
      log.info('Apply droppout with rate {}'.format(self.config.dropout_rate))
      x = tf.nn.dropout(x, self.config.dropout_rate)
    return x
