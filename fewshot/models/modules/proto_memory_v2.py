"""Prototype memory v2. Include support for instrinsic unsupervised loss."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# import numpy as np
import tensorflow as tf

from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.registry import RegisterModule
from fewshot.models.variable_context import variable_scope
from fewshot.models.modules.batch_lru_storage import BatchLRUStorage
# from fewshot.models.modules.diff_utils import round_st
# from fewshot.models.modules.diff_utils import gumbel_softmax

from fewshot.utils.logger import get as get_logger

log = get_logger()

LOGINF = 1e6


@RegisterModule('proto_memory_v2')
class ProtoMemoryV2(ContainerModule):

  def __init__(self, name, dim, config, dtype=tf.float32):
    # Declare a prototype memory for regular classification purpose.
    # Declare another example memory for unsupervised learning purpose.
    super(ProtoMemoryV2, self).__init__(dtype=dtype)
    self._config = config
    self._max_classes = config.max_classes
    self._dim = dim
    radius_init = config.radius_init
    # unknown_logits = config.unknown_logits
    radius_init_write = config.radius_init_write
    with variable_scope(name):
      self._beta = self._get_variable("beta",
                                      self._get_constant_init([], radius_init))
      self._gamma = self._get_variable("gamma", self._get_constant_init([],
                                                                        1.0))
      self._beta2 = self._get_variable(
          "beta2", self._get_constant_init([], radius_init_write))
      self._gamma2 = self._get_variable("gamma2",
                                        self._get_constant_init([], 1.0))
      self._sigma = 1.0
      K = self.max_classes
      self._storage = BatchLRUStorage(
          config.max_bsize,
          K,
          self._dim,
          decay=self.config.decay,
          dtype=self.dtype)

      # Temperature.
      if config.similarity in ["cosine"]:
        self._temp = self._get_variable(
            "temp",
            self._get_constant_init([], config.temp_init),
            trainable=config.temp_learnable)

      if config.use_variables:
        states = self._storage.get_states()
        self._state_var_list = []
        for i, s in enumerate(states):
          self._state_var_list.append(
              self._get_variable(
                  "state_{}".format(i), lambda: s, trainable=False))

  def forward(self,
              x,
              y,
              t,
              *states,
              new_dist=None,
              threshold=0.5,
              ssl_store=None,
              always_new=False,
              newtemp=1.0,
              is_training=True,
              **kwargs):
    # Augment a historical input here.
    y_ = self.retrieve(x, t, *states, beta=new_dist)  # [B, K]
    # tf.print('y1', y_[..., -1], summarize=100)

    if self.config.use_ssl_beta_gamma_write:
      beta2 = self._beta2
      gamma2 = self._gamma2
    else:
      log.info('Not using separate beta gamma for SSL')
      beta2 = None
      gamma2 = None

    if new_dist is not None:
      beta2 = new_dist

    s_shape = [s.shape for s in states]
    y2_ = self.retrieve(x, t, *states, beta=beta2, gamma=gamma2)  # [B, K]
    create = self.config.create_unk
    y_unk, states = self.store(
        x,
        y,
        *states,
        y_pred=y2_,
        ssl_store=ssl_store,
        create_unk=create,
        always_new=always_new,
        threshold=threshold,
        temp=newtemp,
        is_training=is_training)

    for s, shape in zip(states, s_shape):
      s.set_shape(shape)

    if create:
      return y2_, y_unk, states
    else:
      return y_, y_unk, states
    # return y_, states

  def compute_logits(self, x, prototypes):
    """[B, 1, D] x [B, K, D] => [B, K]"""
    if self.config.similarity == "euclidean":
      dist = tf.reduce_sum(tf.square(x - prototypes), [-1])  # [B, K+1]
      return -dist
    elif self.config.similarity == "cosine":
      eps = 1e-5
      p = prototypes
      x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), [-1], keepdims=True))
      p_norm = tf.sqrt(tf.reduce_sum(tf.square(p), [-1], keepdims=True))
      x_ = x / (x_norm + eps)
      p_ = p / (p_norm + eps)
      x_dot_p = tf.matmul(p_, tf.transpose(x_, [0, 2, 1]))[:, :, 0]  # [B, K]
      return x_dot_p * self._temp

  def compute_logits_batch(self, x, prototypes):
    """[B, N, D] x [B, K, D] => [B, N, K]"""
    if self.config.similarity == "euclidean":
      x2 = tf.reduce_sum(tf.square(x), [-1])  # [B, N]
      p2 = tf.reduce_sum(tf.square(prototypes), [-1])  # [B, K]
      xp = tf.matmul(x, tf.transpose(prototypes, [0, 2, 1]))  # [B, N, K]
      return -x2[:, :, None] - p2[:, None, :] + 2 * xp
    elif self.config.similarity == "cosine":
      eps = 1e-5
      p = prototypes
      x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), [-1], keepdims=True))
      p_norm = tf.sqrt(tf.reduce_sum(tf.square(p), [-1], keepdims=True))
      x_ = x / (x_norm + eps)
      p_ = p / (p_norm + eps)
      x_dot_p = tf.matmul(x_, tf.transpose(p_, [0, 2, 1]))  # [B, N, K]
      return x_dot_p * self._temp

  def retrieve(self,
               x,
               t,
               *states,
               write=False,
               beta=None,
               gamma=None,
               **kwargs):
    """Retrieve an item from the prototype memory.

    Args:
      x: [B, D] Inputs.
    """
    # Trim the storage to match shape.
    B = tf.shape(x)[0]
    if tf.less(B, tf.shape(states[0])[0]):
      states = tuple([s[:B] for s in states])

    x = tf.expand_dims(x, 1)  # [B, 1, D]
    prototypes = states[0]  # [B, M, D]
    logits = self.compute_logits(x, prototypes)
    # tf.print('logits', tf.reduce_mean(logits), tf.reduce_max(logits),
    #          tf.reduce_min(logits), tf.shape(logits))

    # Unknown
    if write:
      beta_ = self._beta2
      gamma_ = self._gamma2
    else:
      if beta is None:
        beta_ = self._beta
      else:
        beta_ = beta
      if gamma is None:
        gamma_ = self._gamma
      else:
        gamma_ = gamma

    if self.config.mixture_weight:
      sigma = self._sigma
      count = tf.maximum(states[1], 1e-5)
      count_total = tf.reduce_sum(count, [1], keepdims=True)
      weight = count / count_total
      logits = logits / 2.0 / sigma + tf.math.log(weight)

    # Masking out unused slots.
    valid = tf.greater(states[1], 0.0)  # [B, M]
    logits = tf.where(valid, logits, -LOGINF)  # [B, M]

    if self.config.min_dist:
      log_unk_score = (
          -tf.reduce_max(tf.stop_gradient(logits), [1], keepdims=True) -
          beta_) / gamma_  # [B]
      # log_unk_score = (
      #     -tf.reduce_max(logits, [1], keepdims=True) - beta_) / gamma_  # [B]
    else:
      log_unk_score = (-tf.reduce_logsumexp(
          tf.stop_gradient(logits), [1], keepdims=True) - beta_) / gamma_

    # tf.print('unk', log_unk_score, summarize=100)
    return tf.concat([logits, log_unk_score], axis=-1)  # [B, K+1]

  def store(self,
            x,
            y,
            *states,
            y_pred=None,
            ssl_store=None,
            create_unk=False,
            always_new=False,
            threshold=0.5,
            temp=1.0,
            is_training=True,
            **kwargs):
    """Store items in the prototype memory.

    Args:
      x: [B, D] Inputs.
      y: [B] Label, from the external environment.
      states: Memory states.
      y_pred: [B, M] Logits of prediction, used for learning unlabeled.
      ssl_store: [B] Whether to store unlabeled data.
      create_unk: Bool. Whether to create an unknown cluster.
      temp: Used by straight-through estimator.
    """
    # storage = BatchLRUStorage.from_states(*states, decay=self.config.decay)
    storage = self._storage
    storage.set_states(*states)
    K = storage._storage.shape[1]

    # # Manual decay --> new!!!
    # storage.decay()

    # Store y, if it is known.
    known = 1.0 - tf.cast(tf.equal(y, K), self.dtype)  # [B]
    storage.write_sparse(y, x, mask=known)

    # Store unknown.
    if y_pred is not None:
      if not always_new:
        y_unk_soft = tf.math.sigmoid(y_pred[:, -1])

        # --------------------- Old ---------------------
        if not is_training:
          y_unk = tf.cast(tf.greater(y_unk_soft, threshold), self.dtype)
        else:
          if self.config.straight_through:
            # --------------------- Relaxed straight through ----------------
            import tensorflow_probability as tfp
            dist = tfp.distributions.RelaxedBernoulli(
                temp, logits=y_pred[:, -1])
            y_unk = dist.sample()
            y_unk_hard = tf.cast(tf.greater(y_unk, threshold), self.dtype)
            y_unk = y_unk + y_unk_hard - tf.stop_gradient(y_unk)
          else:
            y_unk = tf.cast(tf.greater(y_unk_soft, threshold), self.dtype)
      else:
        y_unk_soft = tf.ones_like(y_pred[:, -1])
        y_unk = y_unk_soft

      if self.config.straight_through_softmax:
        # ------------------------------
        # new trick.
        import tensorflow_probability as tfp
        dist = tfp.distributions.RelaxedOneHotCategorical(
            temp, logits=y_pred[:, :-1])
        y_smax = dist.sample()
        y_smax_hard = tf.one_hot(
            tf.argmax(y_smax, axis=-1),
            tf.shape(y_pred)[1] - 1)
        # straight-through
        y_smax = y_smax + y_smax_hard - tf.stop_gradient(y_smax)
      else:
        y_smax = tf.nn.softmax(y_pred[:, :-1], axis=-1)

      unk = tf.cast(tf.equal(y, K), self.dtype)  # [B]

      # Try to remove this with straight-through estimator.
      if self.config.straight_through:
        # known_key = unk[:, None] * y_smax * tf.stop_gradient(
        #     (1.0 - y_unk_soft[:, None]))
        known_key = unk[:, None] * y_smax * tf.stop_gradient(
            (1.0 - y_unk_soft[:, None]))
      else:
        known_key = unk[:, None] * y_smax * (1.0 - y_unk_soft[:, None])
      create_flag = tf.cast(tf.constant(create_unk), self.dtype)  # []
      new_key_mask = y_unk * unk * create_flag  # [B]
      if ssl_store is not None:
        ssl_store_ = tf.cast(ssl_store, self.dtype)
        new_key_mask *= ssl_store_  # [B]
      known_key = known_key * (1.0 - new_key_mask[:, None])  # [B, M]
      new_key = storage.allocate_new_key(
          mask=new_key_mask, one_hot=True)  # [B,M]
      key = known_key + new_key  # [B, M]
      if ssl_store is not None:
        key = key * ssl_store_[:, None]  # [B, M]

      # Pad zeros
      B = tf.shape(x)[0]
      if tf.less(B, tf.shape(states[0])[0]):
        # k_zero = tf.zeros([B - 1, states[1].shape[1]])
        k_zero = tf.zeros([B - 1, tf.shape(states[1])[1]])
        key = tf.concat([key, k_zero], axis=0)  # [B, M]
        # x_zero = tf.zeros([B - 1, x.shape[1]])  # [B-1, D]
        x_zero = tf.zeros([B - 1, tf.shape(x)[1]])  # [B-1, D]
        x = tf.concat([x, x_zero], axis=0)  # [B, D]

      # denom = tf.maximum(storage._usage.all + 1.0, 1e-6)  # [B, M]
      # in_gate = (key / denom)[:, :, None]  # [B, M, 1]
      # forget_gate = 1.0 - in_gate
      # # New!!!!
      # storage.write_dense(key, x, in_gate=in_gate, forget_gate=forget_gate)
      storage.write_dense(key, x)

    # Manual decay ### Old!!!
    storage.decay()
    # tf.print(
    #     'count',
    #     tf.reduce_sum(
    #         tf.cast(tf.greater(storage.usage.all[0], 0.0), tf.int32)),
    #     y_unk,
    #     unk,
    #     new_key_mask,
    #     create_flag,
    #     tf.reduce_sum(new_key),
    #     tf.reduce_sum(known_key),
    #     summarize=100)
    # Allocate new keys.
    return y_unk, storage.get_states()

  def get_initial_state(self, bsize):
    """Get initial states."""
    if self.config.use_variables:
      # self.clear_state()
      return tuple(self._state_var_list)  # Do not clear here!
    else:
      # assert False
      K = self.max_classes
      # storage = BatchLRUStorage(
      #     bsize, K, self._dim, decay=self.config.decay, dtype=self.dtype)

      # TODO: try return the class and I doubt tensorflow graph mode is going
      # to let me.
      self._storage.reset_states()
      return self._storage.get_states()

  def clear_state(self):
    if self.config.use_variables:
      for s in self._state_var_list:
        s.assign(tf.zeros_like(s))

  @property
  def max_classes(self):
    """Maximum number of classes."""
    return self._max_classes

  @property
  def dim(self):
    """Storage dimensionality."""
    return self._dim

  @property
  def config(self):
    """Config."""
    return self._config
