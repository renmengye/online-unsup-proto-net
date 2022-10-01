"""Online prototypical network. This one uses sigmoid probability to indicate
unknowns.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
# import horovod.tensorflow as hvd

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
# from fewshot.models.nets.online_unsup_proto_net import OnlineUnsupProtoNet
from fewshot.models.registry import RegisterModel
from fewshot.models.modules.mlp import MLP
from fewshot.utils.logger import get as get_logger
# from fewshot.utils.dummy_context_mgr import dummy_context_mgr as dcm

log = get_logger()


@RegisterModel("online_siamese_proto_net_v6")
class OnlineSiameseProtoNetV6(OnlineUnsupProtoNet):
  """A memory network that keeps updating the prototypes."""

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OnlineSiameseProtoNetV6, self).__init__(
        config, backbone, memory, distributed=distributed, dtype=dtype)
    dim = memory.dim
    proj_nlayer = self.config.memory_net_config.cluster_projection_nlayer
    self._projector = self.build_projection_head("projector", proj_nlayer, dim,
                                                 dim, dim)
    self._predictor = self.build_projection_head("predictor", proj_nlayer, dim,
                                                 dim, dim)

  def build_projection_head(self,
                            name,
                            nlayer,
                            in_dim,
                            mid_dim,
                            out_dim,
                            dtype=tf.float32):
    """Head for projecting hiddens fo contrastive loss."""
    if nlayer == 0:
      return tf.identity
    else:
      dim_list = [in_dim] + [mid_dim] * nlayer + [out_dim]
      # assert use_bn
      decoder = MLP(
          name,
          dim_list,
          add_bias=True,
          bias_init=None,
          act_func=[tf.nn.relu] * nlayer,
          normalization="batch_norm",
          dtype=dtype)
    return decoder

  def run_head(self, x, head, is_training=tf.constant(True)):
    """Run backbone.

    Args:
      x: [B, T, ...] B: mini-batch, T: episode length.
      is_training: Bool. Whether in training mode.
    Returns:
      h: [B, T, D] D: feature length.
    """
    proj_nlayer = self.config.memory_net_config.cluster_projection_nlayer
    if proj_nlayer > 0:
      x_shape = tf.shape(x)
      new_shape = tf.concat([[x_shape[0] * x_shape[1]], x_shape[2:]], axis=0)
      x = tf.reshape(x, new_shape)
      x = head(x, is_training=is_training)
      h_shape = tf.shape(x)
      old_shape = tf.concat([x_shape[:2], h_shape[1:]], axis=0)
      x = tf.reshape(x, old_shape)
    return x

  @tf.function
  def forward(self,
              x,
              y,
              *states,
              dist_q=None,
              flag=None,
              is_training=True,
              **kwargs):
    """Make a forward pass.
    Args:
      x: [B, T, ...]. Support examples at each timestep.
      y: [B, T]. Support labels at each timestep, note that the label is not
                 revealed until the next step.
      states: Memory states tuple.

    Returns:
      y_pred: [B, T, K+1], Logits at each timestep.
      states: Memory states tuple (updated).
      loss_int: Intrinsic loss function (prior to seeing groundtruth labels).
    """
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    # H = self.backbone.config.height
    # W = self.backbone.config.width
    C = self.backbone.config.num_channels

    if is_training:
      if self.backbone.config.data_format == 'NCHW':
        C_all = x.shape[2]
        M = C_all // C - 1
        x_list = tf.split(x, M + 1, axis=2)  # M+1 x [B, T, C, H, W]
      else:
        C_all = x.shape[-1]
        M = C_all // C - 1
        x_list = tf.split(x, M + 1, axis=-1)  # M+1 x [B, T, H, W, C]
      x = tf.concat(x_list, axis=0)  # [MxB, H, W, C]

    # ------ REGULAR VERSION -------
    h = self.run_backbone(x, is_training=is_training)  # [3xB, D]
    if is_training:
      h_aug = h[B:]  # [M xB, T, D]
      h = h[:B]  # [B, T, D]

    # ------ STOP GRADIENT VERSION -------
    # # log.info('Stop gradient on main branch')
    # h = self.run_backbone(x[:B], is_training=is_training)
    # h_aug = self.run_backbone(x[B:], is_training=is_training)  # [3xB, D]
    # # h_aug = tf.stop_gradient(h_aug)  # SimSiam-like setup.
    # # h = tf.stop_gradient(h)  # SimSiam-like setup.

    y_pred = tf.TensorArray(self.dtype, size=T)
    y_pred_aug = tf.TensorArray(self.dtype, size=T)
    y_pred_lb = tf.TensorArray(self.dtype, size=T)
    if len(states) == 0:
      states = self.memory.get_initial_state(B)

    if self.config.ssl_store_schedule:
      log.info("Using probabilistic semisupervised store schedule")
      store_prob = tf.compat.v1.train.piecewise_constant(
          self._step, list(self.config.ssl_store_step_list),
          list(self.config.ssl_store_prob_list))
      ssl_store = tf.less(tf.random.uniform([B, T], 0.0, 1.0), store_prob)
      self._ssl_store = ssl_store
    else:
      ssl_store = tf.ones([B, T], dtype=tf.bool)

    if dist_q is not None:
      T_ = tf.reduce_min(tf.reduce_sum(flag, [1]))  # [B, T] => [B]
      h_sel = h[:, :T_]
      Tf = tf.cast(T_, tf.float32)
      pdist = -self.memory.compute_logits_batch(h_sel, h_sel)  # [B, T, T]
      pdist = tf.reshape(pdist, [B, -1])
      pdist_sort = tf.sort(pdist, axis=-1)  # [B, T*T]
      idx = tf.minimum(
          tf.cast(tf.math.floor(dist_q * Tf * Tf), tf.int64),
          tf.cast(T_ * T_ - 1, tf.int64))
      new_dist = pdist_sort[:, idx]  # New cluster threshold.
    else:
      new_dist = None

    if flag is None:
      flag = tf.ones([B, T], dtype=self.dtype)
    else:
      flag = tf.cast(flag, self.dtype)

    def _expand(a, b):
      if len(b.shape) == 3:
        return a[:, None, None]
      elif len(b.shape) == 2:
        return a[:, None]
      elif len(b.shape) == 1:
        return a

    flag_sum = tf.reduce_sum(flag, [1])  # [B]
    for t in tf.range(T):
      h_ = self.slice_time(h, t)  # [B, ...]
      y_ = self.slice_time(y, t)  # [B]
      ssl_store_ = ssl_store[:, t]
      y_pred_, states_new = self.memory(
          h_, y_, t, *states, new_dist=new_dist, ssl_store=ssl_store_)
      y_pred = y_pred.write(t, y_pred_)

      # Select valid states.
      states = tuple([
          tf.where(_expand(tf.cast(flag[:, t], tf.bool), s), s_new, s)
          for s, s_new in zip(states, states_new)
      ])
      # states = tuple([tf.stop_gradient(s) for s in states])

      if is_training:
        y_pred_lb_ = self.memory.retrieve(h_, t, *states, write=True)
        y_pred_lb = y_pred_lb.write(t, y_pred_lb_)  # [B, K]

        haug_ = tf.split(self.slice_time(h_aug, t), M, axis=0)
        # M x [B, K]
        y_pred_aug_ = [
            self.memory.retrieve(haug_[i], t, *states, write=True)
            for i in range(M)
        ]
        y_pred_aug_ = tf.stack(y_pred_aug_, axis=0)  # [M, B, K]
        y_pred_aug = y_pred_aug.write(t, y_pred_aug_)

    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])

    if self.config.memory_net_config.use_variables:
      for s, s_var in zip(states, self.memory._state_var_list):
        s_var.assign(s)

    if is_training:
      # [T, M, B, K] -> [B, M, T, K]
      y_pred_aug = tf.transpose(y_pred_aug.stack(), [2, 1, 0, 3])
      y_pred_aug = tf.reshape(y_pred_aug, [B, M * T, -1])  # [B, MT, K]
      y_pred_lb = tf.transpose(y_pred_lb.stack(), [1, 0, 2])  # [B, T, K]

      if self.config.memory_net_config.soft_label:
        y_pred_lb = tf.stop_gradient(y_pred_lb[:, :, :-1])  # [B, T, K]
        y_pred_lb = tf.reshape(
            tf.tile(y_pred_lb[:, None, :, :], [1, M, 1, 1]), [B, M * T, -1])
      else:
        y_pred_lb = tf.argmax(y_pred_lb[:, :, :-1], axis=-1)  # [B, T]
        y_pred_lb = tf.reshape(
            tf.tile(y_pred_lb[:, None, :], [1, M, 1]), [B, M * T])
      flag_all = tf.reshape(tf.tile(flag[:, None, :], [1, M, 1]), [B, M * T])
      online_loss, online_metric = self.compute_online_loss(
          y_pred_lb, y_pred_aug, flag_all)

      h_aug1 = h_aug[:B]  # [B, T, D]
      h_aug1 = self.run_head(h_aug1, self._projector, is_training=is_training)
      h_aug1 = self.run_head(h_aug1, self._predictor, is_training=is_training)
      h_aug2 = h_aug[B:]
      h_aug2 = self.run_head(h_aug2, self._projector, is_training=is_training)
      siam_loss = self.online_siam_loss(h_aug1, tf.stop_gradient(h_aug2), flag)
      online_loss += self.config.memory_net_config.siam_loss * siam_loss

      online_stats = self.compute_online_stats(states)
      for k in online_stats:
        online_metric[k] = online_stats[k]
      online_metric["loss/siam_loss"] = siam_loss
    else:
      online_loss = tf.constant(0.0)
      online_metric = {}

    return y_pred, states, online_loss, online_metric

  def online_siam_loss(self, h1, h2, flag):
    if flag is None:
      flag = tf.ones(tf.shape(h1)[:-1], dtype=self.dtype)  # [B, MT]
    else:
      flag = tf.cast(flag, self.dtype)  # [B, MT]
    flag_sum = tf.reduce_sum(flag, [1])  # [B]
    h1 = tf.math.l2_normalize(h1, axis=-1)
    h2 = tf.math.l2_normalize(h2, axis=-1)
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.reduce_sum(tf.square(h1 - h2), [-1]) * flag, [1]) /
        flag_sum)
    return loss

  def compute_online_stats(self, states):
    size = states[1]  # [B, M]
    age = states[2]  # [B, M]
    count = tf.reduce_sum(tf.cast(tf.greater(size, 0.0), tf.float32),
                          [1])  # [B]
    avgsize = tf.reduce_mean(tf.reduce_sum(size, [1]) / count)  # []
    maxsize = tf.reduce_max(size)  # []
    avgage = tf.reduce_mean(
        tf.reduce_sum(tf.cast(age, tf.float32), [1]) / count)  # []
    maxage = tf.reduce_max(age)  # []
    return {
        'proto memory/size avg': avgsize,
        'proto memory/size max': maxsize,
        'proto memory/age avg': avgage,
        'proto memory/age max': maxage
    }

  def compute_online_loss(self, labels, logits, flag):
    """Loss at each timestep.
    Args:
      labels: [B, MT]
      logits: [B, MT, K]
      flag: [B, MT]
    """
    if flag is None:
      flag = tf.ones(tf.shape(labels), dtype=self.dtype)  # [B, MT]
    else:
      flag = tf.cast(flag, self.dtype)  # [B, MT]
    flag_sum = tf.reduce_sum(flag, [1])  # [B]
    logits = logits[:, :, :-1]
    K = tf.shape(logits)[-1]
    if not self.config.memory_net_config.soft_label:
      labels = tf.one_hot(labels, K, axis=-1)  # [B, MT, K]
    xent = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, axis=-1)  # [B, 2T]
    xent = tf.reduce_mean(tf.reduce_sum(xent * flag, [1]) / flag_sum)
    correct = tf.cast(
        tf.equal(tf.argmax(logits, axis=-1), tf.argmax(labels, axis=-1)),
        tf.float32)
    acc = tf.reduce_mean(tf.reduce_sum(correct * flag, [1]) / flag_sum) * 100.0
    return xent, {
        'loss/ssup contrastive': xent,
        'loss/ssup contrastive acc': acc
    }
