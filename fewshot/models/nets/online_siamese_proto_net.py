"""Online prototypical network. This one uses sigmoid probability to indicate
unknowns

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.nets.episode_recurrent_sigmoid_net import EpisodeRecurrentSigmoidNet  # NOQA
from fewshot.models.nets.online_proto_net import OnlineProtoNet
from fewshot.models.modules.mlp import MLP
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger
# from fewshot.utils.dummy_context_mgr import dummy_context_mgr as dcm

log = get_logger()


@RegisterModel("online_siamese_proto_net")
class OnlineSiameseProtoNet(OnlineProtoNet):
  """A memory network that keeps updating the prototypes."""

  def __init__(self,
               config,
               backbone,
               memory,
               distributed=False,
               dtype=tf.float32):
    super(OnlineSiameseProtoNet, self).__init__(config,
                                                backbone,
                                                memory,
                                                distributed=distributed,
                                                dtype=dtype)
    out_dim = memory.dim
    in_dim = backbone.get_output_dimension()[0]
    proj_nlayer = self.config.memory_net_config.cluster_projection_nlayer
    hid_dim = self.config.contrastive_net_config.decoder_hidden_dim
    self._projector = self.build_projection_head("projector", proj_nlayer,
                                                 in_dim, hid_dim, out_dim)

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
      cfg = self.config.contrastive_net_config
      norm_last = cfg.normalize_last_projector_layer
      if cfg.projector_last_relu:
        act_func = [tf.nn.relu] * (nlayer + 1)
      else:
        act_func = [tf.nn.relu] * nlayer
      decoder = MLP(name,
                    dim_list,
                    add_bias=True,
                    bias_init=None,
                    act_func=act_func,
                    normalization=self.backbone.config.normalization,
                    norm_last_layer=norm_last,
                    act_last_layer=cfg.projector_last_relu,
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
              threshold=-1.0,
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
    if threshold < 0:
      threshold = self.config.memory_net_config.new_cluster_thresh

    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    C = self.backbone.config.num_channels

    if is_training:
      # with augmented views.
      if self.backbone.config.data_format == 'NCHW':
        C_all = x.shape[2]
        M = C_all // C - 1
        x_list = tf.split(x, M + 1, axis=2)  # M+1 x [B, T, C, H, W]
      else:
        C_all = x.shape[-1]
        M = C_all // C - 1
        x_list = tf.split(x, M + 1, axis=-1)  # M+1 x [B, T, H, W, C]

      if M == 2:
        # x = tf.concat(x_list[1:], axis=0)
        x = tf.concat(x_list, axis=0)  # [MxB, H, W, C]
        M = 1
      else:
        x = tf.concat(x_list, axis=0)  # [MxB, H, W, C]

    h = self.run_backbone(x, is_training=is_training)  # [3xB, D]
    # tf.print('h', tf.reduce_mean(h), tf.reduce_max(h), tf.reduce_min(h),
    #          tf.shape(h))

    h = self.run_head(h, self._projector, is_training=is_training)
    # tf.print('h proj', tf.reduce_mean(h), tf.reduce_max(h), tf.reduce_min(h),
    #          tf.shape(h))
    if is_training:
      h_aug = h[B:]  # [M xB, T, D]
      h = h[:B]  # [B, T, D] # Use the first dim

    y_pred = tf.TensorArray(self.dtype, size=T)
    y_unk = tf.TensorArray(self.dtype, size=T)
    y_pred_aug = tf.TensorArray(self.dtype, size=T)
    y_pred_lb = tf.TensorArray(self.dtype, size=T)
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

    if dist_q is not None:
      T_ = tf.reduce_min(tf.reduce_sum(flag, [1]))  # [B, T] => [B]
      h_sel = h[:, :T_]
      Tf = tf.cast(T_, tf.float32)
      pdist = -self.memory.compute_logits_batch(h_sel, h_sel)  # [B, T, T]
      pdist = tf.reshape(pdist, [B, -1])
      pdist_sort = tf.sort(pdist, axis=-1)  # [B, T*T]
      idx = tf.minimum(tf.cast(tf.math.floor(dist_q * Tf * Tf), tf.int64),
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
    K = self.config.memory_net_config.max_classes
    sim = tf.zeros([K, K])
    selector = tf.cast(1 - tf.eye(K), tf.bool)
    for t in tf.range(T):
      h_ = self.slice_time(h, t)  # [B, ...]
      y_ = self.slice_time(y, t)  # [B]
      ssl_store_ = ssl_store[:, t]
      # ---------------- Begin New ----------------
      # y_pred_ = self.memory.retrieve(h_, t, *states, write=True)
      # ---------------- End New ------------------
      # ---------------- Begin Old ----------------
      # Old after storage.
      newtemp = 2.0 - tf.minimum(tf.cast(self._step, tf.float32),
                                 20000.0) / 20000.0 * 1.7
      # newtemp = 5.0 - tf.minimum(tf.cast(self._step, tf.float32),
      #                            40000.0) / 40000.0 * 4.9
      y_pred_, y_unk_, states_new = self.memory(
          h_,
          y_,
          t,
          *states,
          new_dist=new_dist,
          threshold=threshold,
          newtemp=newtemp,
          ssl_store=ssl_store_,
          is_training=is_training,
          always_new=self.config.memory_net_config.always_new)

      # Record inter-cluster statistics
      memory_items = self.memory._storage.all  # [B, K, D]
      sim = self.memory.compute_logits_batch(memory_items,
                                             memory_items)[0]  # [K, K]

      y_pred = y_pred.write(t, y_pred_)
      y_unk = y_unk.write(t, y_unk_)

      # Select valid states.
      states = tuple([
          tf.where(_expand(tf.cast(flag[:, t], tf.bool), s), s_new, s)
          for s, s_new in zip(states, states_new)
      ])

      # ---------------- End Old ----------------
      if is_training:
        y_pred_lb_ = self.memory.retrieve(h_, t, *states, write=True)
        y_pred_lb = y_pred_lb.write(t, y_pred_lb_)  # [B, K]

        haug_ = tf.split(self.slice_time(h_aug, t), M, axis=0)
        # New aug scheme
        # haug_ = [self.slice_time(h_aug, t)]
        # M x [B, K]
        y_pred_aug_ = [
            self.memory.retrieve(haug_[i], t, *states, write=True)
            for i in range(M)
        ]
        y_pred_aug_ = tf.stack(y_pred_aug_, axis=0)  # [M, B, K]
        y_pred_aug = y_pred_aug.write(t, y_pred_aug_)

      # # ---------------- Begin New ----------------
      # # New before storage!!!
      # y_pred_, states_new = self.memory(
      #     h_,
      #     y_,
      #     t,
      #     *states,
      #     new_dist=new_dist,
      #     threshold=threshold,
      #     ssl_store=ssl_store_)
      # y_pred = y_pred.write(t, y_pred_)

      # # Select valid states.
      # states = tuple([
      #     tf.where(_expand(tf.cast(flag[:, t], tf.bool), s), s_new, s)
      #     for s, s_new in zip(states, states_new)
      # ])
      # # ---------------- End New ----------------

    y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
    y_unk = tf.transpose(y_unk.stack(), [1, 0])

    if self.config.memory_net_config.use_variables:
      for s, s_var in zip(states, self.memory._state_var_list):
        s_var.assign(s)

    if is_training:
      # [T, M, B, K] -> [B, M, T, K]
      y_pred_aug = tf.transpose(y_pred_aug.stack(), [2, 1, 0, 3])
      y_pred_aug = tf.reshape(y_pred_aug, [B, M * T, -1])  # [B, MT, K]
      y_pred_lb = tf.transpose(y_pred_lb.stack(), [1, 0, 2])  # [B, T, K]
      y_pred_lb = tf.reshape(tf.tile(y_pred_lb[:, None, :, :], [1, M, 1, 1]),
                             [B, M * T, -1])
      y_pred_lb = tf.stop_gradient(y_pred_lb)  # [B, T, K]
      # if self.config.memory_net_config.soft_label:
      #   # Old
      #   # y_pred_lb = tf.stop_gradient(y_pred_lb[:, :, :-1])  # [B, T, K]
      #   # New before entropy
      #   # y_pred_lb = tf.nn.softmax(y_pred_lb, axis=-1)
      # else:
      #   y_pred_lb = tf.argmax(y_pred_lb[:, :, :-1], axis=-1)  # [B, T]
      #   y_pred_lb = tf.reshape(
      #       tf.tile(y_pred_lb[:, None, :], [1, M, 1]), [B, M * T])
      flag_all = tf.reshape(tf.tile(flag[:, None, :], [1, M, 1]), [B, M * T])
      loss, metric = self.compute_ssup_loss(y_pred, y_unk, y_pred_lb,
                                            y_pred_aug, flag_all)
      stats = self.compute_stats(states)
      sim_ = sim[selector]
      stats["cluster sim/max"] = tf.reduce_max(sim_)
      stats["cluster sim/mean"] = tf.reduce_mean(sim_)
      stats["cluster sim/min"] = tf.reduce_min(sim_)
      for k in stats:
        metric[k] = stats[k]
    else:
      loss = tf.constant(0.0)
      metric = {}
    return y_pred, y_pred_lb, states, loss, metric

  def compute_stats(self, states):
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

  def renormalize(self, logits):
    """Renormalize the logits, with the last dimension to be unknown. The rest
    of the softmax is then multiplied by 1 - sigmoid."""
    logits_unk = logits[:, :, -1:]
    logits_rest = logits[:, :, :-1]
    logits_rest = tf.math.log_softmax(logits_rest)
    logits_unk_c = tf.math.log(1.0 - tf.math.sigmoid(logits_unk) + 1e-7)
    logits_rest += logits_unk_c
    return tf.concat([logits_rest, tf.math.log_sigmoid(logits_unk)], axis=-1)

  def compute_ssup_loss(self, pred, pred_unk, labels, logits, flag):
    """Loss at each timestep.
    Args:
      pred: [B, MT, K+1], prediction, before storage.
      labels: [B, MT] (hard) or [B, MT, K] (soft), labels, after storage.
      logits: [B, MT, K+1], logits from augmentation, after storage.
      flag: [B, MT] (binary)
    """
    # 1. Self-supervision loss.
    logits_new = logits[:, :, -1]
    labels_new = labels[:, :, -1]
    if self.config.memory_net_config.soft_label:
      labels_new = tf.math.sigmoid(labels_new)
    else:
      labels_new = tf.cast(labels_new > 0.0, tf.float32)
    logits_old = logits[:, :, :-1]
    labels_old = labels[:, :, :-1]
    K = tf.shape(logits_old)[-1]
    if self.config.memory_net_config.soft_label:
      temp = self.config.memory_net_config.soft_label_temp
      labels_old = tf.nn.softmax(labels_old / temp, axis=-1)  # [B, MT, K]
    else:
      labels_old = tf.one_hot(tf.argmax(labels_old, axis=-1), K,
                              axis=-1)  # [B, MT, K]

    if flag is None:
      # [B, MT] or [B, MT, K]
      flag = tf.ones(tf.shape(labels_old), dtype=self.dtype)
    else:
      flag = tf.cast(flag, self.dtype)  # [B, MT]
    flag_sum = tf.reduce_sum(flag, [1])  # [B]
    # if not self.config.memory_net_config.soft_label:
    #   labels = tf.one_hot(labels_old, K, axis=-1)
    xent = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_old, logits=logits_old, axis=-1)  # * (1.0 - labels_new)
    xent = tf.reduce_mean(tf.reduce_sum(xent * flag, [1]) / flag_sum)
    correct = tf.cast(
        tf.equal(tf.argmax(logits_old, axis=-1), tf.argmax(labels, axis=-1)),
        tf.float32)
    acc = tf.reduce_mean(tf.reduce_sum(correct * flag, [1]) / flag_sum) * 100.0

    # 2. Entropy loss.
    pred_old = pred[:, :, :-1]
    pred_new = pred[:, :, -1]
    new_prob_soft = tf.math.sigmoid(pred_new)
    new_prob0 = pred_unk

    y_smax = tf.nn.softmax(pred_old, axis=-1)  # [B, T, K]
    y_hard = tf.one_hot(tf.argmax(pred_old, axis=-1),
                        tf.shape(pred)[-1] - 1,
                        axis=-1)  # [B, T, K]

    T_ = tf.cast(tf.reduce_min(flag_sum), tf.int64)

    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_smax,
                                                      logits=pred_old,
                                                      axis=-1)
    # if self.config.memory_net_config.soft_label:
    #   # Old
    #   entropy = tf.nn.softmax_cross_entropy_with_logits(
    #       labels=y_smax, logits=pred_old, axis=-1)
    #   #  * tf.stop_gradient(
    #   # 1.0 - new_prob0)  # [B, T]

    # else:
    #   # Old
    #   entropy = tf.nn.softmax_cross_entropy_with_logits(
    #       labels=y_hard, logits=pred_old, axis=-1) * tf.stop_gradient(
    #           1.0 - new_prob0)  # [B, T]

    entropy = tf.reduce_sum(entropy * flag, [1]) / flag_sum  # [B]
    entropy = tf.reduce_mean(entropy)

    # 3. New cluster loss.
    # tf.print("new prob", new_prob0)
    new_prob = new_prob0 * flag  # [B, T]
    # new_prob = new_prob0
    T_ = tf.cast(tf.reduce_min(flag_sum), tf.int64)
    new_prob_max = tf.reduce_max(new_prob_soft[:, :T_])  # []
    new_prob_min = tf.reduce_min(new_prob_soft[:, :T_])  # []
    new_prob = tf.reduce_sum(new_prob, [1]) / flag_sum  # [B]
    new_prob = tf.reduce_mean(new_prob)  # []
    # tf.print("new prob", new_prob)
    decay_steps = self.config.memory_net_config.linear_schedule
    target_mean = self.config.memory_net_config.new_prob_target
    if decay_steps > 0:
      coeff = tf.minimum(
          tf.cast(self._step, self.dtype) / float(decay_steps), 1.0)
      target_mean = coeff * target_mean + 0.7 * (1.0 - coeff)

    import tensorflow_probability as tfp
    # SUM = 4.0
    # alpha = SUM * target_mean
    # beta = SUM * (1.0 - target_mean)

    SUM = 4.0
    alpha = (SUM - 2.0) * target_mean + 1.0  # actually, mode
    beta = SUM - alpha
    # print(alpha, beta)
    # assert False
    # alpha = 8.0 * target_mean
    # beta = 8.0 * (1.0 - target_mean)
    one = 1.0 - 1e-3 - tf.stop_gradient(new_prob) + new_prob
    zero = 1e-3 - tf.stop_gradient(new_prob) + new_prob
    new_prob_clip = tf.clip_by_value(new_prob, zero, one)
    nc_loss = -tfp.distributions.Beta(alpha, beta).log_prob(new_prob_clip)
    # nc_loss = 0.5 * tf.math.abs(new_prob - target_mean)
    # tf.print('nc loss', alpha, beta, nc_loss)
    nc_entropy = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=new_prob0, logits=pred_new) * flag, [1]) / flag_sum
    nc_entropy = tf.reduce_mean(nc_entropy)
    nc_loss_w = self.config.memory_net_config.new_cluster_loss
    nc_ent_w = self.config.memory_net_config.new_cluster_entropy_loss
    ent_w = self.config.memory_net_config.entropy_loss

    # tf.print('xent', xent, 'entropy', entropy, 'nc loss', nc_loss,
    # 'new prob', new_prob)
    loss = xent + ent_w * entropy + nc_loss_w * nc_loss + nc_ent_w * nc_entropy

    return loss, {
        'loss/ssup contrastive': xent,
        'loss/ssup contrastive acc': acc,
        'loss/ssup entropy': entropy,
        'loss/ssup new prob': new_prob,
        'loss/ssup new prob max': new_prob_max,
        'loss/ssup new prob min': new_prob_min,
        'loss/ssup new entropy': nc_entropy,
        'loss/ssup new thresh': target_mean
    }

  def compute_loss(self, logits, y_gt, flag=None, **kwargs):
    return 0.0, {}
