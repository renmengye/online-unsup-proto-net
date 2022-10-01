"""Online prototypical network. This one uses sigmoid probability to indicate
unknowns.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
# from fewshot.models.nets.online_proto_net import OnlineProtoNet
from fewshot.models.modules.mlp import MLP
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger
# from fewshot.utils.dummy_context_mgr import dummy_context_mgr as dcm
from fewshot.models.modules.batch_storage import BatchStorage

log = get_logger()


@RegisterModel("online_swav_queue_net")
class OnlineSwavQueueNet(EpisodeRecurrentSigmoidNet):
  """A memory network that keeps updating the prototypes."""

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OnlineSwavQueueNet, self).__init__(
        config, backbone, distributed=distributed, dtype=dtype)
    # self._dim = backbone.get_output_dimension()[-1]
    D = backbone.get_output_dimension()[-1]
    # Note: this does not have any projection layers.
    # D = config.contrastive_net_config.output_dim
    self._memory = memory
    dim = memory.dim
    K = self.config.contrastive_net_config.num_prototypes
    proj_nlayer = self.config.contrastive_net_config.decoder_nlayer
    self._projector = self.build_projection_head("projector", proj_nlayer, dim,
                                                 dim, dim)
    self._prototypes = self._get_variable("prototypes",
                                          self._get_normal_init([D, K]))
    B = config.optimizer_config.batch_size
    Q = config.contrastive_net_config.queue_size
    self._queue = BatchStorage(B, Q, [D])

  def sinkhorn_batch(self, scores, eps=0.05, niters=3):
    """scores: [B, T, K]"""
    # [B, T, K] -> [B, K, T]
    # eps=0.05
    Q = tf.transpose(tf.math.exp(scores / eps), [0, 2, 1])
    Q /= tf.reduce_sum(Q, [1], keepdims=True)  # [B, K, T] / [B, T]
    B = tf.shape(Q)[0]
    K = tf.shape(Q)[1]
    T = tf.shape(Q)[2]
    # print(B, K, T)
    u = tf.zeros([B, K])  # [B, K]
    r = tf.ones([B, K, 1]) / tf.cast(K, tf.float32)  # [B, K, 1]
    c = tf.ones([B, 1, T]) / tf.cast(T, tf.float32)  # [B, 1, T]
    for _ in range(niters):
      u = tf.reduce_sum(Q, [2], keepdims=True)  # [B, K, T] -> [B, K, 1]
      Q *= (r / u)  # [B, K, T] x [B, K, 1]
      Q *= (c / tf.reduce_sum(Q, [1], keepdims=True))  # [B, K, T] x [B, 1, T]
    # [B, T, K]
    return tf.transpose(Q / tf.reduce_sum(Q, [1], keepdims=True), [0, 2, 1])

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

  def regularized_weights(self):
    """List of weights to be L2 regularized"""
    rw = super(OnlineSwavQueueNet, self).regularized_weights()
    for excl in ['head_supervised', 'prototypes']:
      rw = list(filter(lambda x: excl not in x.name, rw))
    # print(rw)
    # assert False
    return rw

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
      assert M == 2, str(M)

    Q = self.config.contrastive_net_config.queue_size
    warmup_step = self.config.contrastive_net_config.queue_warmup_steps

    # ------ REGULAR VERSION -------
    h = self.run_backbone(x, is_training=is_training)  # [3xB, D]
    if is_training:
      h_aug = h[B:]  # [M xB, T, D]
      h = h[:B]  # [B, T, D]

      # if self._step < warmup_step:
      #   ssup_label = tf.zeros([B, M * T])  # [B, MT]
      #   ssup_pred = tf.one_hot(
      #       ssup_label,
      #       depth=self.config.contrastive_net_config.num_prototypes)
      # else:
      # ------ STOP GRADIENT VERSION -------
      hidden = self.run_head(h_aug, self._projector, is_training=True)
      if self.config.contrastive_net_config.hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)

      hidden_q = tf.reshape(self._queue.all, [B, Q, -1])
      hidden1 = hidden[:B, :vT]
      hidden2 = hidden[B:2 * B, :vT]  # [B, T, D]

      if self._step > warmup_step:
        hidden1q = tf.concat([hidden_q, hidden1], axis=1)  # [B, Q+T, D]
        hidden2q = tf.concat([hidden_q, hidden2], axis=1)  # [B, Q+T, D]
      else:
        hidden1q = hidden1
        hidden2q = hidden2

      # [B, D] x [D, K] = [B, K]
      temperature = self.config.contrastive_net_config.temperature
      # Normalize
      proto_norm = self._prototypes / tf.math.sqrt(
          tf.reduce_sum(tf.math.square(self._prototypes), [0]) + 1e-5)
      self._prototypes.assign(proto_norm)
      # [B, T, D] x [D, K] = [B, T, K]
      score1 = tf.matmul(hidden1q, self._prototypes)
      score2 = tf.matmul(hidden2q, self._prototypes)
      q1 = self.sinkhorn_batch(score1)
      q2 = self.sinkhorn_batch(score2)
      q1 = tf.stop_gradient(q1)
      q2 = tf.stop_gradient(q2)
      logits1 = score1 / temperature
      logits2 = score2 / temperature

      # Take the current batch part.
      if self._step >= warmup_step:
        q1 = q1[:, Q:]
        q2 = q2[:, Q:]
        logits1 = logits1[:, Q:]
        logits2 = logits2[:, Q:]
      ssup_label = tf.concat([q2, q1], axis=1)  # [B, T, K]
      ssup_pred = tf.concat([logits1, logits2], axis=1)  # [B, T, K]

      # Store examples to the queue.
      T64 = tf.cast(T, tf.int64)
      item_idx = tf.reshape(
          tf.tile(
              tf.reshape(
                  tf.math.mod(
                      tf.range(self._step * T64, (self._step + 1) * T64), Q),
                  [1, -1]), [B, 1]), [-1])
      b_idx = tf.reshape(
          tf.tile(tf.reshape(tf.range(B, dtype=tf.int64), [-1, 1]), [1, T]),
          [-1])
      key_batch = tf.stack([b_idx, item_idx], axis=-1)
      self._queue.setitem_batch(key_batch, tf.reshape(hidden[:B],
                                                      [B * T, -1]))  # [BT, 2]

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
      online_loss, online_metric = self.compute_online_loss(
          ssup_label, ssup_pred, None)
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

  def compute_loss(self, logits, y_gt, flag=None, **kwargs):
    """Compute the training loss."""
    reg_loss = self._get_regularizer_loss(*self.regularized_weights())
    # tf.print(self.regularized_weights())
    # tf.print(reg_loss)
    return reg_loss * self.wd, {}

  def compute_online_loss(self, labels, logits, flag):
    """Loss at each timestep.
    Args:
      labels: [B, MT, K]
      logits: [B, MT, K]
      flag: [B, MT]
    """
    if flag is None:
      flag = tf.ones(tf.shape(labels)[:-1], dtype=self.dtype)  # [B, MT]
    else:
      flag = tf.cast(flag, self.dtype)  # [B, MT]
    flag_sum = tf.reduce_sum(flag, [1])  # [B]
    # logits = logits[:, :, :-1]
    K = tf.shape(logits)[-1]
    xent = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, axis=-1)  # [B, 2T]
    xent = tf.reduce_mean(tf.reduce_sum(xent * flag, [1]) / flag_sum)
    correct = tf.cast(
        tf.equal(tf.argmax(logits, axis=-1), tf.argmax(labels, axis=-1)),
        tf.float32)
    acc = tf.reduce_mean(tf.reduce_sum(correct * flag, [1]) / flag_sum)
    return xent, {
        'loss/ssup contrastive': xent,
        'loss/ssup contrastive acc': acc * 100.0
    }

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
    self.memory.clear_state()
    r = super(OnlineSwavQueueNet, self).eval_step(x, y, **kwargs)
    self.memory.clear_state()
    return r

  def train_step(self, x, y, y_gt, flag, writer, **kwargs):
    results = super().train_step(
        x, y, y_gt=y_gt, flag=flag, writer=writer, **kwargs)

    self._prototypes.assign(self._prototypes / tf.math.sqrt(
        tf.reduce_sum(tf.math.square(self._prototypes), [0])))
    return results

  @property
  def memory(self):
    """Memory module"""
    return self._memory
