"""Online prototypical network. This one uses sigmoid probability to indicate
unknowns.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
import horovod.tensorflow as hvd

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
from fewshot.models.nets.online_siamese_proto_net import OnlineSiameseProtoNet
from fewshot.models.nets.online_proto_net import OnlineProtoNet
from fewshot.models.modules.mlp import MLP
from fewshot.models.modules.proto_memory_v2 import ProtoMemoryV2
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger
from fewshot.utils.dummy_context_mgr import dummy_context_mgr as dcm

log = get_logger()


@RegisterModel("online_siamese_proto_net_v4")
class OnlineSiameseProtoNetV4(OnlineSiameseProtoNet):
  """A memory network that keeps updating the prototypes."""

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OnlineSiameseProtoNetV4, self).__init__(
        config, backbone, memory, distributed=distributed, dtype=dtype)
    self._memory_e = ProtoMemoryV2(
        "proto_memory_e", memory.dim, memory.config, dtype=tf.float32)
    dim = memory.dim
    proj_nlayer = self.config.memory_net_config.cluster_projection_nlayer
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
      decoder = MLP(
          name,
          dim_list,
          add_bias=True,
          bias_init=None,
          act_func=[tf.nn.relu] * nlayer,
          normalization="batch_norm",
          dtype=dtype)
    return decoder

  def run_projector(self, x, is_training=tf.constant(True)):
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
      x = self._projector(x, is_training=is_training)
      h_shape = tf.shape(x)
      old_shape = tf.concat([x_shape[:2], h_shape[1:]], axis=0)
      x = tf.reshape(x, old_shape)
    return x

  # @tf.function
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
    h = self.run_backbone(x, is_training=is_training)  # [3xB, D]
    if is_training:
      x = h[:B]  # [B, T, D]
      x_aug = h[B:]  # [M xB, T, D]
    else:
      x = h

    # Projection for cluster disambiguition.
    if self.config.memory_net_config.cluster_projection_nlayer > 0:
      h_proj = self.run_projector(h, is_training=is_training)
      x_proj = h_proj[:B]
      x_aug_proj = h_proj[B:]
    else:
      x_proj = x
      x_aug_proj = x_aug

    # Cluster memory prediction.
    y_pred = tf.TensorArray(self.dtype, size=T)
    y_pred_aug = tf.TensorArray(self.dtype, size=T)
    y_pred_lb = tf.TensorArray(self.dtype, size=T)

    # Example memory prediction.
    y_pred_aug_e = tf.TensorArray(self.dtype, size=T)
    y_pred_lb_e = tf.TensorArray(self.dtype, size=T)

    if len(states) == 0:
      states = self.memory.get_initial_state(B)
      states_e = self.memory_e.get_initial_state(B)
      cold_start = True
    else:
      nstate = len(states)
      states_e = states[nstate // 2:]
      states = states[:nstate // 2]
      cold_start = False

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
      x_sel = x_proj[:, :T_]
      Tf = tf.cast(T_, tf.float32)
      pdist = -self.memory.compute_logits_batch(x_sel, x_sel)  # [B, T, T]
      pdist = tf.reshape(pdist, [B, -1])
      pdist_sort = tf.sort(pdist, axis=-1)  # [B, T*T]
      idx = tf.cast(tf.math.floor(dist_q * Tf * Tf), tf.int64)
      new_dist = pdist_sort[:, idx]  # New cluster threshold.
      # tf.print('q', dist_q, 'beta', new_dist)
    else:
      new_dist = None
      # assert False

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
      xp_ = self.slice_time(x_proj, t)  # [B, ...]
      x_ = self.slice_time(x, t)  # [B, ...]
      y_ = self.slice_time(y, t)  # [B]
      ssl_store_ = ssl_store[:, t]

      # Cluster memory.
      y_pred_, states_new = self.memory(
          xp_, y_, t, *states, new_dist=new_dist, ssl_store=ssl_store_)
      y_pred = y_pred.write(t, y_pred_)

      # Example memory.
      _, states_new_e = self.memory_e(
          x_, y_, t, *states_e, create_new=True, always_new=True)

      # Select valid states.
      states = tuple([
          tf.where(_expand(tf.cast(flag[:, t], tf.bool), s), s_new, s)
          for s, s_new in zip(states, states_new)
      ])
      states_e = tuple([
          tf.where(_expand(tf.cast(flag[:, t], tf.bool), s), s_new, s)
          for s, s_new in zip(states_e, states_new_e)
      ])

      if is_training:
        # Cluster memory.
        y_pred_lb = y_pred_lb.write(t,
                                    self.memory.retrieve(
                                        xp_, t, *states, write=True))  # [B, K]
        xaugp_ = tf.split(self.slice_time(x_aug_proj, t), M, axis=0)
        # M x [B, K]
        y_pred_aug_ = [
            self.memory.retrieve(xaugp_[i], t, *states, write=True)
            for i in range(M)
        ]
        # [M, B, K]
        y_pred_aug = y_pred_aug.write(t, tf.stack(y_pred_aug_, axis=0))

        # Example memory.
        y_pred_lb_e = y_pred_lb_e.write(
            t, self.memory_e.retrieve(x_, t, *states, write=True))
        xaug_ = tf.split(self.slice_time(x_aug, t), M, axis=0)
        # M x [B, K]
        y_pred_aug_e_ = [
            self.memory_e.retrieve(xaug_[i], t, *states, write=True)
            for i in range(M)
        ]
        # [M, B, K]
        y_pred_aug_e = y_pred_aug_e.write(t, tf.stack(y_pred_aug_e_, axis=0))

    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])

    if self.config.memory_net_config.use_variables:
      for s, s_var in zip(states, self.memory._state_var_list):
        s_var.assign(s)

    if is_training:
      # Cluster loss.
      # [T, M, B, K] -> [B, M, T, K]
      y_pred_aug = tf.transpose(y_pred_aug.stack(), [2, 1, 0, 3])
      y_pred_aug = tf.reshape(y_pred_aug, [B, M * T, -1])  # [B, MT, K]
      y_pred_lb = tf.transpose(y_pred_lb.stack(), [1, 0, 2])  # [B, T, K]
      y_pred_lb = tf.argmax(y_pred_lb[:, :, :-1], axis=-1)  # [B, T]
      y_pred_lb = tf.reshape(
          tf.tile(y_pred_lb[:, None, :], [1, M, 1]), [B, M * T])
      flag_all = tf.reshape(tf.tile(flag[:, None, :], [1, M, 1]), [B, M * T])
      online_loss, online_metric = self.compute_online_loss(
          y_pred_lb, y_pred_aug, flag_all)
      online_stats = self.compute_online_stats(states)
      for k in online_stats:
        online_metric[k] = online_stats[k]

      # Example loss.
      y_pred_aug_e = tf.transpose(y_pred_aug_e.stack(), [2, 1, 0, 3])
      y_pred_aug_e = tf.reshape(y_pred_aug_e, [B, M * T, -1])  # [B, MT, K]
      y_pred_lb_e = tf.transpose(y_pred_lb_e.stack(), [1, 0, 2])  # [B, T, K]
      y_pred_lb_e = tf.argmax(y_pred_lb_e[:, :, :-1], axis=-1)  # [B, T]
      y_pred_lb_e = tf.reshape(
          tf.tile(y_pred_lb_e[:, None, :], [1, M, 1]), [B, M * T])
      online_loss_e, online_metric_e = self.compute_online_loss(
          y_pred_lb_e, y_pred_aug_e, flag_all)
      for k in online_metric_e:
        online_metric[k + ' ex'] = online_metric_e[k]

      lambda_e = self.config.memory_net_config.example_contrastive_loss
      online_loss += lambda_e * online_loss_e  # TODO tune the coefficients.
    else:
      online_loss = tf.constant(0.0)

    return y_pred, states + states_e, online_loss, online_metric

  @property
  def memory_e(self):
    return self._memory_e

  def eval_step(self, x, y, **kwargs):
    """One evaluation step.
    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep.

    Returns:
      logits: [B, T, Kmax], prediction.
    """
    self.memory_e.clear_state()
    r = super(OnlineSiameseProtoNetV4, self).eval_step(x, y, **kwargs)
    self.memory_e.clear_state()
    return r
