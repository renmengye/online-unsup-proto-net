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


@RegisterModel("swav_net")
class SWAVNet(Net):

  def __init__(self,
               config,
               backbone,
               num_train_examples,
               distributed=False,
               dtype=tf.float32):
    super(SWAVNet, self).__init__()
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
    K = self.config.contrastive_net_config.num_prototypes
    self._prototypes = self._get_variable(
        "prototypes",
        self._get_normal_init([config.contrastive_net_config.output_dim, K]))

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
      return super(SWAVNet, self)._get_optimizer(optname, learn_rate)

  def sinkhorn(self, scores, eps=0.05, niters=3):
    Q = tf.transpose(tf.math.exp(scores / eps))  # [B, K] -> [K, B]
    Q /= tf.reduce_sum(Q, [0], keepdims=True)  # [K, B] -> [B]
    K, B = Q.shape
    K = tf.shape(Q)[0]
    B = tf.shape(Q)[1]
    u = tf.zeros([K])  # [K]
    r = tf.ones([K]) / tf.cast(K, tf.float32)  # [K]
    c = tf.ones([B]) / tf.cast(B, tf.float32)  # [B]
    for _ in range(niters):
      u = tf.reduce_sum(Q, [1])  # [K, B] -> [K]
      Q *= (r / u)[:, None]  # [K, B] x [K, 1]
      Q *= (c / tf.reduce_sum(Q, [0], keepdims=True))  # [K, B] x [1, B]
    return tf.transpose(Q / tf.reduce_sum(Q, [0], keepdims=True))

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
      return tf.compat.v1.train.piecewise_constant(
          self.step, list(np.array(decay_steps).astype(np.int64)),
          list(self.config.optimizer_config.lr_list))

  def forward(self, x, is_training=True):
    """Run forward pass."""
    h = self.backbone(x, is_training=is_training)
    hidden = self._decoder(h, is_training=is_training)
    return h, hidden

  def loss_hidden(self, hidden1, hidden2):
    """Computes the loss function based on hidden activations."""
    batch_size = tf.shape(hidden1)[0]
    temperature = self.config.contrastive_net_config.temperature
    weights = 1.0
    # [B, D] x [D, K] = [B, K]
    score1 = tf.matmul(hidden1, self._prototypes)
    score2 = tf.matmul(hidden2, self._prototypes)

    # Gather hidden1/hidden2 across replicas and create local labels.
    if self._distributed:
      score1_large = hvd.allgather(score1, name='score1')
      score2_large = hvd.allgather(score2, name='score2')
      enlarged_batch_size = tf.shape(score1_large)[0]
    else:
      score1_large = score1
      score2_large = score2

    # TODO: Add queue implementation (optional).
    q1 = self.sinkhorn(score1_large)
    q2 = self.sinkhorn(score2_large)
    q1 = tf.stop_gradient(q1)
    q2 = tf.stop_gradient(q2)

    # p1 = tf.nn.softmax(q1 / temperature)
    # p2 = tf.nn.softmax(q2 / temperature)

    LARGE_NUM = 1e9

    logits1 = score1_large / temperature
    logits2 = score2_large / temperature
    loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=q2, logits=logits1)
    loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=q1, logits=logits2)
    loss = 0.5 * tf.reduce_mean(loss1 + loss2)

    q1h = tf.argmax(q1, axis=1)  # [B]
    q2h = tf.argmax(q2, axis=1)  # [B]

    acc1 = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(logits1, -1, output_type=q2h.dtype), q2h),
            tf.float32))
    acc2 = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(logits2, -1, output_type=q1h.dtype), q1h),
            tf.float32))
    acc = 0.5 * (acc1 + acc2)
    return loss, acc

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
    rw = super(SWAVNet, self).regularized_weights()
    for excl in ['head_supervised', 'prototypes']:
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

      # [D, K]
      self._prototypes.assign(self._prototypes / tf.math.sqrt(
          tf.reduce_sum(tf.math.square(self._prototypes), [0])))
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
