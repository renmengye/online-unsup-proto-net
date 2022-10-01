"""SIMCLR. A network with a backbone encoder and a decoder.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from fewshot.data.preprocessors.simclr_utils import batch_random_blur
from fewshot.models.modules.mlp import MLP
from fewshot.models.nets.net import Net
from fewshot.models.optimizers import LARSOptimizer
from fewshot.models.registry import RegisterModel
from fewshot.utils.dummy_context_mgr import dummy_context_mgr
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("simclr_net")
class SIMCLRNet(Net):

  def __init__(self,
               config,
               backbone,
               num_train_examples,
               distributed=False,
               dtype=tf.float32):
    super(SIMCLRNet, self).__init__()
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
    norm_last = config.contrastive_net_config.normalize_last_projector_layer
    self._decoder = self.build_projection_head(
        "head_contrastive",
        config.contrastive_net_config.decoder_nlayer,
        self._dim,
        config.contrastive_net_config.decoder_hidden_dim,
        config.contrastive_net_config.output_dim,
        norm_last_layer=norm_last,
        dtype=dtype)
    self._distributed = distributed

  def _get_optimizer(self, optname, learn_rate):
    if optname == 'lars':
      return LARSOptimizer(
          learn_rate,
          weight_decay=self.wd,
          name="LARSOptimizer",
          exclude_from_weight_decay=[
              'bn', 'batch_norm', '/b', 'head_supervised'
          ])
    else:
      return super(SIMCLRNet, self)._get_optimizer(optname, learn_rate)

  def build_projection_head(self,
                            name,
                            nlayer,
                            in_dim,
                            mid_dim,
                            out_dim,
                            norm_last_layer=False,
                            dtype=tf.float32):
    """Head for projecting hiddens fo contrastive loss."""
    if nlayer == 0:
      return tf.identity
    else:
      dim_list = [in_dim] + [mid_dim] * nlayer + [out_dim]
      normalization = self.config.contrastive_net_config.decoder_normalization
      # assert use_bn
      decoder = MLP(
          name,
          dim_list,
          add_bias=True,
          bias_init=None,
          act_func=[tf.nn.relu] * nlayer,
          normalization=normalization,
          norm_last_layer=norm_last_layer,
          dtype=dtype)
    return decoder

  def cosine_decay(self,
                   learning_rate,
                   global_step,
                   decay_steps,
                   alpha=0.0,
                   name=None):
    global_step = tf.cast(global_step, tf.float32)
    decay_steps = tf.cast(decay_steps, tf.float32)
    global_step = tf.minimum(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed
    return decayed_learning_rate

  @property
  def learn_rate(self):
    """Build learning rate schedule."""
    # Cosine decay.
    # if self.config.optimizer_config.optimizer == 'lars':
    if self.config.optimizer_config.learn_rate_schedule == 'cosine':

      def _learn_rate():
        base_learning_rate = self.config.optimizer_config.lr_list[0]
        num_examples = self._num_train_examples
        global_step = self.step
        batch_size = self.global_batch_size
        lr_scaling = self.config.optimizer_config.lr_scaling
        warmup_epochs = self.config.optimizer_config.warmup_epochs
        warmup_steps = int(round(warmup_epochs * num_examples // batch_size))
        if lr_scaling == 'linear':
          scaled_lr = base_learning_rate * batch_size / 256.
        elif lr_scaling == 'sqrt':
          scaled_lr = base_learning_rate * np.sqrt(batch_size / 256.)
        elif lr_scaling == 'none':
          scaled_lr = base_learning_rate
        else:
          raise ValueError(
              'Unknown learning rate scaling {}'.format(lr_scaling))
        learning_rate = (tf.cast(global_step, tf.float32) / int(warmup_steps) *
                         scaled_lr if warmup_steps else scaled_lr)
        # Cosine decay learning rate schedule
        # total_steps = self.config.optimizer_config.max_train_steps
        total_epochs = self.config.optimizer_config.max_train_epochs
        total_steps = total_epochs * int(np.ceil(num_examples / batch_size))
        learning_rate = tf.where(
            global_step < warmup_steps, learning_rate,
            self.cosine_decay(scaled_lr, global_step - warmup_steps,
                              total_steps - warmup_steps))
        return learning_rate

      return _learn_rate
    elif self.config.optimizer_config.learn_rate_schedule == 'staircase':
      # Regular staircase (based on epoch number).
      num_examples = self._num_train_examples
      batch_size = self.global_batch_size
      decay_steps = [
          int(round(e * num_examples // batch_size))
          for e in self.config.optimizer_config.lr_decay_epochs
      ]
      lr_list = [float(lr) for lr in self.config.optimizer_config.lr_list]
      return tf.compat.v1.train.piecewise_constant(self.step, decay_steps,
                                                   lr_list)

  def forward(self, x, is_training=True):
    """Run forward pass."""
    h = self.backbone(x, is_training=is_training)
    if self.config.contrastive_net_config.decoder_nlayer > 0:
      hidden = self._decoder(h, is_training=is_training)
    else:
      hidden = h
    return h, hidden

  def loss_hidden(self, hidden1, hidden2):
    """Computes the loss function based on hidden activations."""
    batch_size = tf.shape(hidden1)[0]
    temperature = self.config.contrastive_net_config.temperature
    weights = 1.0

    # Gather hidden1/hidden2 across replicas and create local labels.
    if self._distributed:
      hidden1_large = hvd.allgather(hidden1, name='hidden1')
      hidden2_large = hvd.allgather(hidden2, name='hidden2')
      enlarged_batch_size = tf.shape(hidden1_large)[0]
      labels_idx = tf.range(batch_size) + hvd.rank() * batch_size
      labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
      masks = tf.one_hot(labels_idx, enlarged_batch_size)
    else:
      hidden1_large = hidden1
      hidden2_large = hidden2
      labels_idx = tf.range(batch_size)
      labels = tf.one_hot(labels_idx, batch_size * 2)
      masks = tf.one_hot(labels_idx, batch_size)

    LARGE_NUM = 1e9
    logits_aa = tf.matmul(
        hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(
        hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(
        hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(
        hidden2, hidden1_large, transpose_b=True) / temperature

    logits_a = tf.concat([logits_ab, logits_aa], 1)
    logits_b = tf.concat([logits_ba, logits_bb], 1)
    loss_a = tf.compat.v1.losses.softmax_cross_entropy(
        labels, logits_a, weights=weights)
    loss_b = tf.compat.v1.losses.softmax_cross_entropy(
        labels, logits_b, weights=weights)
    xent = loss_a + loss_b

    acc_a = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(logits_a, -1, output_type=labels_idx.dtype),
                labels_idx), tf.float32))
    acc_b = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(logits_b, -1, output_type=labels_idx.dtype),
                labels_idx), tf.float32))
    acc = 0.5 * acc_a + 0.5 * acc_b
    return xent, acc

  def loss(self, x, reg=True):
    """Compute loss function."""
    # H = self.backbone.config.height
    # W = self.backbone.config.width
    C = self.backbone.config.num_channels

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
    h, hidden = self.forward(x, is_training=True)  # [2B, C], [2B, D]
    if self.config.contrastive_net_config.hidden_norm:
      hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, axis=0)

    xent, acc = self.loss_hidden(hidden1, hidden2)
    if reg:
      reg_loss = self._get_regularizer_loss(*self.regularized_weights())
      loss = xent + reg_loss * self.wd
    else:
      loss = xent

    result = {
        'xent': xent,
        'contrastive train acc': acc * 100.0,
    }

    # Regularize activation directly.
    act_l1_loss = 0.0
    if self.config.contrastive_net_config.use_l1_reg:
      l1_reg_coeff = self.config.contrastive_net_config.l1_reg_coeff

      # Option 1, before cosine normalization, maybe not much effect.
      # act_l1_loss = l1_reg_coeff * tf.reduce_sum(tf.abs(h))
      # hplus = tf.maximum(h, 0.0)

      # Option 2, after cosine normalization, regularizing both pos + neg.
      hplus = tf.abs(hidden)
      act_l1_loss = l1_reg_coeff * 10.0 * tf.reduce_mean(
          hplus * tf.math.exp(-3.0 * hplus))
      loss += act_l1_loss
      result['act l1 loss'] = act_l1_loss

    # Difference in activation between two views.
    if self.config.contrastive_net_config.use_l1_diff:
      l1_diff_coeff = self.config.contrastive_net_config.l1_diff_coeff
      h1, h2 = tf.split(h, 2, axis=0)
      diff_l1_loss = l1_diff_coeff * tf.reduce_sum(tf.abs(h1 - h2))
      loss += diff_l1_loss
      result['diff l1 loss'] = diff_l1_loss

    return loss, result

  def _get_regularizer_loss(self, *w_list):
    """Computes L2 loss."""
    if len(w_list) > 0:
      return tf.add_n([tf.reduce_sum(w**2) * 0.5 for w in w_list])
    else:
      return 0.0

  def regularized_weights(self):
    """List of weights to be L2 regularized"""
    rw = super(SIMCLRNet, self).regularized_weights()
    for excl in ['head_supervised']:
      rw = list(filter(lambda x: excl not in x.name, rw))
    return rw

  @tf.function
  def train_step(self, x, writer=None, **kwargs):
    """One training step."""
    with writer.as_default() if writer is not None else dummy_context_mgr():
      # Calculates gradients
      with tf.GradientTape() as tape:
        # LARS have regularizer built-in.
        if self.config.optimizer_config.optimizer == 'lars':
          reg = False
        else:
          reg = True
        loss, metric = self.loss(x, reg=reg)

      # apply gradients.
      if self._distributed:
        tape = hvd.DistributedGradientTape(tape)
        xent = metric['xent']
        xent = tf.reduce_mean(
            hvd.allgather(tf.zeros([1]) + metric['xent'], name="loss"))
      else:
        xent = metric['xent']

      var_list = self.var_to_optimize()
      grad_list = tape.gradient(loss, var_list)
      opt = self.optimizer

      if self.config.optimizer_config.optimizer == 'lars':
        opt.apply_gradients(zip(grad_list, var_list), global_step=self.step)
      else:
        self._step.assign(self._step + 1)
        opt.apply_gradients(zip(grad_list, var_list))
      if tf.equal(tf.math.mod(self.step, 10), 0) and writer is not None:
        for k in metric:
          if k != 'xent':
            tf.summary.scalar(k, metric[k], step=self._step + 1)
    return xent

  @property
  def step(self):
    return self._step

  @property
  def backbone(self):
    return self._backbone

  @property
  def decoder(self):
    return self._decoder

  @property
  def wd(self):
    return self._wd

  @property
  def config(self):
    return self._config

  @property
  def global_batch_size(self):
    return self._global_batch_size
