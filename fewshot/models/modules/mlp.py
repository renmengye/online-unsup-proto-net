"""Multi-layer perceptron.
Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.modules.layer_norm import LayerNorm
from fewshot.models.modules.nnlib import BatchNorm
from fewshot.models.modules.sync_batch_norm import SyncBatchNorm
from fewshot.models.modules.nnlib import Linear, CosineLinear
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope


@RegisterModule('mlp')
class MLP(ContainerModule):

  def __init__(self,
               name,
               layer_size,
               add_bias=True,
               bias_init=None,
               act_func=None,
               normalization=None,
               norm_last_layer=False,
               act_last_layer=False,
               dtype=tf.float32):
    super(MLP, self).__init__(dtype=dtype)
    self._layers = []
    with variable_scope(name):
      for i in range(len(layer_size) - 1):
        if bias_init is not None:

          def bi():
            return tf.zeros([layer_size[i + 1]], dtype=dtype) + bias_init[i]
        else:
          bi = None
        self._layers.append(
            Linear(
                "layer_{}".format(i),
                layer_size[i],
                layer_size[i + 1],
                b_init=bi,
                add_bias=add_bias,
                dtype=dtype))

        # Normalization.
        if i < len(layer_size) - 2 or (norm_last_layer and
                                       i < len(layer_size) - 1):
          if normalization == 'layer_norm':
            assert False
            self._layers.append(
                LayerNorm(
                    "layernorm_{}".format(i), layer_size[i + 1], dtype=dtype))
          elif normalization == 'batch_norm':
            self._layers.append(
                BatchNorm(
                    "batchnorm_{}".format(i),
                    layer_size[i + 1],
                    dtype=dtype,
                    data_format="NC"))
          elif normalization == 'sync_batch_norm':
            self._layers.append(
                SyncBatchNorm(
                    "batchnorm_{}".format(i),
                    layer_size[i + 1],
                    dtype=dtype,
                    data_format="NC"))

        # Activation function.
        if i < len(layer_size) - 2 or (act_last_layer and
                                       i < len(layer_size) - 1):
          if act_func is None:
            self._layers.append(tf.nn.relu)
          else:
            self._layers.append(act_func[i])

  def forward(self, x, is_training=True):
    """Forward pass."""
    for layer in self._layers:
      if isinstance(layer, BatchNorm) or isinstance(layer, SyncBatchNorm):
        # tf.print('head', tf.shape(x), layer)
        x = layer(x, is_training=is_training)
      else:
        x = layer(x)
    return x


@RegisterModule('cosine-last-mlp')
class CosineLastMLP(ContainerModule):

  def __init__(self,
               name,
               layer_size,
               add_bias=True,
               bias_init=None,
               act_func=None,
               layernorm=False,
               temp=None,
               learn_temp=False,
               dtype=tf.float32,
               wdict=None):
    super(CosineLastMLP, self).__init__(dtype=dtype)
    self._layers = []
    with variable_scope(name):
      for i in range(len(layer_size) - 1):
        if bias_init is not None and bias_init[i] is not None:

          def bi():
            return tf.zeros([layer_size[i + 1]], dtype=dtype) + bias_init[i]
        else:
          bi = None

        if i < len(layer_size) - 2:
          layer = Linear(
              "layer_{}".format(i),
              layer_size[i],
              layer_size[i + 1],
              b_init=bi,
              add_bias=add_bias,
              dtype=dtype,
              wdict=wdict)
        else:
          layer = CosineLinear(
              "layer_{}".format(i),
              layer_size[i],
              layer_size[i + 1],
              temp=temp,
              learn_temp=learn_temp,
              dtype=tf.float32,
              wdict=wdict)
        self._layers.append(layer)
        if layernorm:
          self._layers.append(
              LayerNorm(
                  "layernorm_{}".format(i),
                  layer_size[i + 1],
                  dtype=dtype,
                  wdict=wdict))
        if i < len(layer_size) - 2:
          if act_func is None:
            self._layers.append(tf.nn.relu)
          else:
            self._layers.append(act_func[i])

  def forward(self, x):
    """Forward pass."""
    for layer in self._layers:
      x = layer(x)
    return x
