"""Basic MLP backbone. """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
from fewshot.models.modules.backbone import Backbone
from fewshot.models.modules.mlp import MLP
from fewshot.models.registry import RegisterModule


@RegisterModule("mlp_backbone")
class MLPBackbone(Backbone):

  def __init__(self, config):
    super(MLPBackbone, self).__init__(config)
    self._config = config
    self._mlp = MLP(
        "mlp", [config.num_inputs] + list(config.num_filters),
        add_bias=True,
        bias_init=None,
        act_func=None,
        normalization=config.normalization,
        dtype=tf.float32)

  def get_output_dimension(self):
    return [self.config.num_filters[-1]]

  def forward(self, x, **kwargs):
    x = tf.reshape(x, [-1, self.config.num_inputs])
    return self._mlp(x, **kwargs)

  @property
  def config(self):
    return self._config
