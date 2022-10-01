"""A network for pretraining regular classification tasks.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from fewshot.models.modules.nnlib import Linear
from fewshot.models.nets.net import Net
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("pretrain_net")
class PretrainNet(Net):

  def __init__(self, config, backbone, dtype=tf.float32):
    super(PretrainNet, self).__init__()
    self._backbone = backbone
    self._config = config
    assert self.config.num_classes > 0, 'Must specify number of output classes'
    opt_config = self.config.optimizer_config
    gs = tf.Variable(0, dtype=tf.int64, name='step', trainable=False)
    self._step = gs
    self._wd = backbone.config.weight_decay
    ngpu = hvd.size()
    self._distributed = ngpu > 1
    if self._distributed:
      lr_decay_steps = np.array(opt_config.lr_decay_steps).astype(np.int64)
      lr_decay_steps = lr_decay_steps // ngpu
      learning_rate = tf.compat.v1.train.piecewise_constant(
          self.step, list(lr_decay_steps), list(opt_config.lr_list))

      def _learn_rate():
        scaled_lr = learning_rate() * ngpu
        warmup_steps = self.config.optimizer_config.warmup_steps
        decay = (tf.cast(gs, tf.float32) / int(warmup_steps) *
                 scaled_lr if warmup_steps > 0 else scaled_lr)
        return tf.where(gs < warmup_steps, decay, scaled_lr)

      self._learn_rate = _learn_rate
    else:
      if len(opt_config.lr_decay_steps) > 0:
        lr_decay_steps = np.array(opt_config.lr_decay_steps).astype(np.int64)
        learning_rate = tf.compat.v1.train.piecewise_constant(
            self.step, list(lr_decay_steps), list(opt_config.lr_list))
      else:
        learning_rate = opt_config.lr_list[0]
      self._learn_rate = learning_rate
    opt = self._get_optimizer(opt_config.optimizer, self.learn_rate)
    self._optimizer = opt
    out_dim = backbone.get_output_dimension()
    self._fc = Linear("fc", out_dim[-1], config.num_classes, dtype=dtype)

  def forward(self, x, is_training=tf.constant(True)):
    """Run forward pass."""
    h = self.backbone(x, is_training=is_training)
    # tf.print(is_training, 'max', tf.reduce_max(h), 'min', tf.reduce_min(h))
    logits = self._fc(h)
    return logits

  @tf.function
  def train_step(self, x, y, is_training=True):
    """One training step."""
    # Calculates gradients
    with tf.GradientTape() as tape:
      logits = self.forward(x, is_training=is_training)
      xent = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                         labels=y))
      reg_loss = self._get_regularizer_loss(*self.regularized_weights())
      loss = xent + reg_loss * self.wd
      var_list = self.var_to_optimize()

    # apply gradients.
    if self._distributed:
      tape = hvd.DistributedGradientTape(tape)
      xent = tf.reduce_mean(hvd.allgather(tf.zeros([1]) + xent, name="loss"))
    grad_list = tape.gradient(loss, var_list)
    self._step.assign_add(1)
    self.optimizer.apply_gradients(zip(grad_list, var_list))
    return xent

  @tf.function
  def eval_step(self, x):
    """One evaluation step."""
    prediction = self.forward(x, is_training=tf.constant(False))
    return prediction

  @property
  def learn_rate(self):
    return self._learn_rate

  @property
  def step(self):
    return self._step

  @property
  def backbone(self):
    return self._backbone

  @property
  def wd(self):
    return self._wd

  @property
  def config(self):
    return self._config
