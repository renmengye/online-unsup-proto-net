"""Online prototypical network. This one uses sigmoid probability to indicate
unknowns.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
from fewshot.models.registry import RegisterModel
from fewshot.models.modules.mlp import MLP
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("online_simclr_net")
class OnlineSimclrNet(EpisodeRecurrentSigmoidNet):
  """A memory network that keeps updating the prototypes."""

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OnlineSimclrNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    D = backbone.get_output_dimension()[-1]
    self._memory = memory
    dim = memory.dim
    proj_nlayer = self.config.contrastive_net_config.decoder_nlayer
    self._projector = self.build_projection_head("projector", proj_nlayer, dim,
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
      norm = self.config.contrastive_net_config.decoder_normalization
      decoder = MLP(
          name,
          dim_list,
          add_bias=True,
          bias_init=None,
          act_func=[tf.nn.relu] * nlayer,
          normalization=norm,
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
    proj_nlayer = self.config.contrastive_net_config.decoder_nlayer
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
    C = self.backbone.config.num_channels
    if flag is None:
      flag = tf.ones([B, T], dtype=self.dtype)
    else:
      flag = tf.cast(flag, self.dtype)
    flag_sum = tf.reduce_sum(flag, [1])  # [B]
    vT = tf.cast(tf.reduce_min(flag_sum), tf.int32)

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
      assert M == 2

    h = self.run_backbone(x, is_training=is_training)  # [3xB, D]
    if is_training:
      h_aug = h[B:]  # [M xB, T, D]
      h = h[:B]  # [B, T, D]
      hidden = self.run_head(h_aug, self._projector, is_training=True)
      # tf.print('after project', tf.shape(hidden))
      if self.config.contrastive_net_config.hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
      hidden1, hidden2 = tf.split(hidden[:, :vT], 2, axis=0)  # [B, T, D]

    y_pred = tf.TensorArray(self.dtype, size=T)

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
      # T_ = tf.cast(tf.reduce_min(tf.reduce_sum(flag, [1])), tf.int32)
      h_sel = h[:, :vT]
      Tf = tf.cast(vT, tf.float32)
      pdist = -self.memory.compute_logits_batch(h_sel, h_sel)  # [B, T, T]
      pdist = tf.reshape(pdist, [B, -1])
      pdist_sort = tf.sort(pdist, axis=-1)  # [B, T*T]
      idx = tf.cast(tf.math.floor(dist_q * Tf * Tf), tf.int64)
      new_dist = pdist_sort[:, idx]  # New cluster threshold.
    else:
      new_dist = None

    def _expand(a, b):
      if len(b.shape) == 3:
        return a[:, None, None]
      elif len(b.shape) == 2:
        return a[:, None]
      elif len(b.shape) == 1:
        return a

    if is_training:
      y_pred = tf.zeros([B, T, self.memory.max_classes])
      online_loss, online_metric = self.compute_online_loss(hidden1, hidden2)
    else:
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
      y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
      online_loss, online_metric = 0.0, {}

      if self.config.memory_net_config.use_variables:
        for s, s_var in zip(states, self.memory._state_var_list):
          s_var.assign(s)

    return y_pred, states, online_loss, online_metric

  def compute_online_loss(self, hidden1, hidden2):
    """Loss at each timestep.
    Args:
      labels: [B, T, D]
      logits: [B, T, D]
    """
    assert hidden1.shape[0] == 1
    hidden1 = hidden1[0]
    hidden2 = hidden2[0]
    batch_size = tf.shape(hidden1)[0]
    temperature = self.config.contrastive_net_config.temperature
    weights = 1.0

    if self._distributed:
      import horovod.tensorflow as hvd
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

    xent = 0.5 * loss_a + 0.5 * loss_b
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

    # batch_size = tf.shape(hidden1)[0]
    # labels_idx = tf.range(batch_size)
    # labels = tf.one_hot(labels_idx, batch_size * 2)
    # masks = tf.one_hot(labels_idx, batch_size)
    # temp = self.config.contrastive_net_config.temperature
    # weights = 1.0

    # LARGE_NUM = 1e9
    # logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temp
    # logits_aa = logits_aa - masks * LARGE_NUM
    # logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temp
    # logits_bb = logits_bb - masks * LARGE_NUM
    # logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temp
    # logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temp
    # logits_a = tf.concat([logits_ab, logits_aa], 1)
    # logits_b = tf.concat([logits_ba, logits_bb], 1)
    # loss_a = tf.compat.v1.losses.softmax_cross_entropy(
    #     labels, logits_a, weights=weights)
    # loss_b = tf.compat.v1.losses.softmax_cross_entropy(
    #     labels, logits_b, weights=weights)
    # acc_a = tf.reduce_mean(
    #     tf.cast(
    #         tf.equal(
    #             tf.argmax(logits_a, -1, output_type=labels_idx.dtype),
    #             labels_idx), tf.float32))
    # acc_b = tf.reduce_mean(
    #     tf.cast(
    #         tf.equal(
    #             tf.argmax(logits_b, -1, output_type=labels_idx.dtype),
    #             labels_idx), tf.float32))
    # xent = 0.5 * loss_a + 0.5 * loss_b
    # acc = 0.5 * acc_a + 0.5 * acc_b
    return xent, {
        'loss/ssup contrastive': xent,
        'loss/ssup contrastive acc': acc * 100.0
    }

  def get_metric_init(self):
    return {}

  def compute_loss(self, logits, y_gt, flag=None, **kwargs):
    """Compute the training loss."""
    reg_loss = self._get_regularizer_loss(*self.regularized_weights())
    return reg_loss * self.wd, {}

  # @tf.function
  def eval_step(self, x, y, **kwargs):
    """One evaluation step.
    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep.

    Returns:
      logits: [B, T, Kmax], prediction.
    """
    self.memory.clear_state()
    r = super(OnlineSimclrNet, self).eval_step(x, y, **kwargs)
    self.memory.clear_state()
    return r

  @property
  def memory(self):
    """Memory module"""
    return self._memory
