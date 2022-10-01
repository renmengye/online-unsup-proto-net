"""Online prototypical network. This one uses sigmoid probability to indicate
unknowns.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("online_proto_net")
@RegisterModel("online_proto_sigmoid_net")
@RegisterModel("proto_mem_sigmoid_net")  # Legacy name
class OnlineProtoNet(EpisodeRecurrentSigmoidNet):
  """A memory network that keeps updating the prototypes."""

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OnlineProtoNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    self._memory = memory

  @tf.function
  def forward(self,
              x,
              y,
              *states,
              dist_q=None,
              threshold=0.5,
              flag=None,
              is_training=True,
              **kwargs):
    """Make a forward pass.
    Args:
      x: [B, T, ...]. Support examples at each timestep.
      y: [B, T]. Support labels at each timestep, note that the label is not
                 revealed until the next step.
      states: Memory states tuple.ww
      dist_q: []. Quantile for sweeping thresholds for new cluster.
      flag: [B, T]. Training flag of whether the sequence has ended.
      is_training: Bool.

    Returns:
      y_pred: [B, T, K+1], Logits at each timestep.
      states: Memory states tuple (updated).
      loss_int: Intrinsic loss function (prior to seeing groundtruth labels).
    """
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    x = self.run_backbone(x, is_training=is_training)
    y_pred = tf.TensorArray(self.dtype, size=T)
    if len(states) == 0:
      states = self.memory.get_initial_state(B)
      cold_start = True
    else:
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

    online_loss = tf.zeros([B], dtype=self.dtype)
    online_metric = self.get_metric_init()

    if dist_q is not None:
      # assert x.shape[0] == 1, "Only support at eval time."
      T_ = tf.reduce_min(tf.reduce_sum(flag, [1]))  # [B, T] => [B]
      x_sel = x[:, :T_]
      Tf = tf.cast(T_, tf.float32)
      pdist = -self.memory.compute_logits_batch(x_sel, x_sel)  # [B, T, T]
      pdist = tf.reshape(pdist, [B, -1])
      pdist_sort = tf.sort(pdist, axis=-1)  # [B, T*T]
      idx = tf.cast(tf.math.floor(dist_q * Tf * Tf), tf.int64)
      new_dist = pdist_sort[:, idx]  # New cluster threshold.
    else:
      new_dist = None

    if flag is None:
      flag = tf.ones([B, T], dtype=self.dtype)
    else:
      flag = tf.cast(flag, self.dtype)
    flag_sum = tf.reduce_sum(flag, [1])  # [B]

    def _expand(a, b):
      if len(b.shape) == 3:
        return a[:, None, None]
      elif len(b.shape) == 2:
        return a[:, None]
      elif len(b.shape) == 1:
        return a

    for t in tf.range(T):
      x_ = self.slice_time(x, t)  # [B, ...]
      y_ = self.slice_time(y, t)  # [B]
      ssl_store_ = ssl_store[:, t]
      y_pred_, states_new = self.memory(
          x_,
          y_,
          t,
          *states,
          new_dist=new_dist,
          threshold=threshold,
          ssl_store=ssl_store_)

      # Select valid states.
      states = tuple([
          tf.where(_expand(tf.cast(flag[:, t], tf.bool), s), s_new, s)
          for s, s_new in zip(states, states_new)
      ])

      y_pred = y_pred.write(t, y_pred_)
      loss_, metric_ = self.compute_online_loss(y_pred_, states)

      if loss_ is not None:
        loss_ = loss_ * flag[:, t] / flag_sum
        online_loss += loss_

    online_loss = tf.reduce_mean(online_loss)
    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])

    if self.config.memory_net_config.use_variables:
      for s, s_var in zip(states, self.memory._state_var_list):
        s_var.assign(s)

    return y_pred, None, states, online_loss, online_metric

  def compute_online_loss(self, y_pred, states):
    """Loss at each timestep."""
    return None, None

  def get_metric_init(self):
    return {}

  # @tf.function
  def eval_step(self, x, y, **kwargs):
    """One evaluation step.
    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep.

    Returns:
      logits: [B, T, Kmax], prediction.
    """
    if self.memory.config.use_variables:
      states = [s.numpy() for s in self.memory._state_var_list]
      self.memory.clear_state()
    r = super(OnlineProtoNet, self).eval_step(x, y, **kwargs)
    if self.memory.config.use_variables:
      self.memory.clear_state()
      for i, s in enumerate(states):
        self.memory._state_var_list[i].assign(s)
    return r

  @property
  def memory(self):
    """Memory module"""
    return self._memory
