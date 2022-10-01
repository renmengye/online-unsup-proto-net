"""FILM enabled batch norm module.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.nnlib import BatchNorm
from fewshot.models.modules.group_norm import GroupNorm
from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope


class FilmNorm(ContainerModule):

  def __init__(self, norm, wdict=None):
    super(FilmNorm, self).__init__(norm.dtype)
    self._norm = norm
    if norm._data_format == "NCHW":
      self._axes2 = [0, 1, 3, 4]
    elif norm._data_format == "NHWC":
      self._axes2 = [0, 1, 2, 3]
    with variable_scope(norm._name + '_film'):
      self._beta0 = self._get_variable(
          "beta0", self._get_constant_init([], 0.0), wdict=wdict)
      self._gamma0 = self._get_variable(
          "gamma0", self._get_constant_init([], 0.0), wdict=wdict)

  def _expand2(self, v):
    C = tf.shape(v)[-1]
    return tf.reshape(v, [-1, 1, C, 1, 1])

  def _expand3(self, v):
    C = tf.shape(v)[-1]
    return tf.reshape(v, [-1, 1, 1, 1, C])

  def apply_gamma_beta(self, x, beta_gate, gamma_gate):
    x_shape = tf.shape(x)
    B = tf.shape(beta_gate)[0]
    if self._norm._data_format == "NCHW":
      H, W, C = x_shape[2], x_shape[3], x_shape[1]
      x = tf.reshape(x, [B, -1, C, H, W])
      gamma = self._expand2(gamma_gate) * self._gamma0
      beta = self._expand2(beta_gate) * self._beta0
      gamma += 1.0
    else:
      H, W, C = x_shape[1], x_shape[2], x_shape[3]
      x = tf.reshape(x, [B, -1, H, W, C])
      gamma = self._expand3(gamma_gate) * self._gamma0
      beta = self._expand3(beta_gate) * self._beta0
      gamma += 1.0
    x = x * gamma + beta
    return x


class FilmBatchNorm(FilmNorm):
  """Batch normalization layer."""

  def __init__(self,
               name,
               num_channels,
               data_format="NCHW",
               eps=1e-3,
               decay=0.999,
               dtype=tf.float32,
               wdict=None):
    norm = BatchNorm(
        name,
        num_channels,
        data_format=data_format,
        eps=eps,
        decay=decay,
        dtype=dtype,
        wdict=wdict)
    super(FilmBatchNorm, self).__init__(norm, wdict=wdict)

  def forward(self,
              x,
              beta_gate=None,
              gamma_gate=None,
              is_training=tf.constant(True),
              **kwargs):
    if is_training:
      return self.train_forward(x, beta_gate, gamma_gate)
    else:
      return self.eval_forward(x, beta_gate, gamma_gate)

  def train_forward(self, x, beta_gate=None, gamma_gate=None):
    x = self._norm.train_forward(x)
    if beta_gate is None or gamma_gate is None:
      return x
    x_old = x
    x = self.apply_gamma_beta(x, beta_gate, gamma_gate)
    return tf.reshape(x, x_old.shape)

  def eval_forward(self, x, beta_gate=None, gamma_gate=None):
    x = self._norm.eval_forward(x)
    if beta_gate is None or gamma_gate is None:
      return x
    x_old = x
    x = self.apply_gamma_beta(x, beta_gate, gamma_gate)
    return tf.reshape(x, x_old.shape)


class FilmGroupNorm(FilmNorm):
  """Batch normalization layer."""

  def __init__(self,
               name,
               num_channels,
               num_groups,
               data_format="NCHW",
               eps=1e-3,
               dtype=tf.float32,
               wdict=None):
    norm = GroupNorm(
        name,
        num_channels,
        num_groups,
        data_format=data_format,
        eps=eps,
        dtype=dtype,
        wdict=wdict)
    super(FilmGroupNorm, self).__init__(norm)

  def forward(self, x, beta_gate=None, gamma_gate=None, **kwargs):
    x = self._norm.forward(x, **kwargs)
    if beta_gate is None or gamma_gate is None:
      return x
    x_old = x
    x = self.apply_gamma_beta(x, beta_gate, gamma_gate)
    return tf.reshape(x, x_old.shape)
