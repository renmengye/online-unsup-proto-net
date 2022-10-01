"""SIMCLR. A network with a backbone encoder and a decoder.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.data.preprocessors.simclr_utils import batch_random_blur
from fewshot.models.nets.simclr_net import SIMCLRNet
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("simsiam_net")
class SIMSIAMNet(SIMCLRNet):

  def __init__(self,
               config,
               backbone,
               num_train_examples,
               distributed=False,
               dtype=tf.float32):
    super(SIMSIAMNet, self).__init__(
        config,
        backbone,
        num_train_examples,
        distributed=distributed,
        dtype=dtype)
    self._backbone = backbone
    self._config = config
    assert self.config.num_classes > 0, 'Must specify number of output classes'
    opt_config = self.config.optimizer_config
    gs = tf.Variable(0, dtype=tf.int64, name='step', trainable=False)
    self._step = gs
    self._wd = backbone.config.weight_decay

    if distributed:
      import horovod.tensorflow as hvd
      bsize = opt_config.batch_size * hvd.size()
    else:
      bsize = opt_config.batch_size
    self._global_batch_size = bsize

    self._num_train_examples = num_train_examples
    opt = self._get_optimizer(opt_config.optimizer, self.learn_rate)
    self._optimizer = opt
    self._dim = backbone.get_output_dimension()[-1]
    self._projector = self.build_projection_head(
        "projector",
        config.contrastive_net_config.projector_nlayer,
        self._dim,
        config.contrastive_net_config.projector_hidden_dim,
        config.contrastive_net_config.projector_hidden_dim,
        norm_last_layer=True,
        dtype=dtype)

    # Double check hyperparameters here.
    self._predictor = self.build_projection_head(
        "predictor",
        config.contrastive_net_config.predictor_nlayer,
        config.contrastive_net_config.projector_hidden_dim,
        config.contrastive_net_config.predictor_hidden_dim,
        config.contrastive_net_config.output_dim,
        norm_last_layer=False,  # No BN last layer.
        dtype=dtype)
    self._distributed = distributed

  def forward(self, x, is_training=True):
    """Run forward pass."""
    h = self.backbone(x, is_training=is_training)
    hidden = self._projector(h, is_training=is_training)
    pred = self._predictor(hidden, is_training=is_training)
    return h, hidden, pred

  def loss_hidden(self, hidden, pred):
    """Computes the loss function based on hidden activations."""
    return -tf.reduce_mean(tf.reduce_sum(hidden * pred, [-1]))

  def loss(self, x, reg=True):
    """Compute loss function."""
    if self.backbone.config.data_format == 'NCHW':
      x_list = tf.split(x, 2, axis=1)
      H, W = x.shape[2], x.shape[3]
    else:
      x_list = tf.split(x, 2, axis=-1)
      H, W = x.shape[1], x.shape[2]

    if self.config.simclr_random_blur:
      x_list = batch_random_blur(x_list, H, W)
    x = tf.concat(x_list, axis=0)

    # Hidden activation before and after decoder.
    h, hidden, pred = self.forward(x, is_training=True)  # [2B, C], [2B, D]

    if self.config.contrastive_net_config.hidden_norm:
      hidden = tf.math.l2_normalize(hidden, -1)
      pred = tf.math.l2_normalize(pred, -1)

    hidden1, hidden2 = tf.split(hidden, 2, axis=0)
    pred1, pred2 = tf.split(pred, 2, axis=0)

    # SimSiam specific.
    # loss1 = self.loss_hidden(hidden1, tf.stop_gradient(pred2))
    # loss2 = self.loss_hidden(hidden2, tf.stop_gradient(pred1))
    loss1 = self.loss_hidden(tf.stop_gradient(hidden1), pred2)
    loss2 = self.loss_hidden(tf.stop_gradient(hidden2), pred1)
    loss = 0.5 * loss1 + 0.5 * loss2

    if reg:
      reg_loss = self._get_regularizer_loss(*self.regularized_weights())
      loss = loss + reg_loss * self.wd
    else:
      loss = loss

    result = {'xent': loss}
    return loss, result

  @property
  def projector(self):
    return self._projector

  @property
  def predictor(self):
    return self._predictor
