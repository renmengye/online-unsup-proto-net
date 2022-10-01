"""Contextual Prototypical Memory: ProtoNet memory plus an RNN module.
Note that this is the semisupervised version.
This should be compatible to fully supervised sequence as well.

Use the RNN module to
  1) encode the examples (with context);
  2) output unknown.

Use the ProtoNet to
  1) decode the class;
  2) store class prototypes.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
from fewshot.models.registry import RegisterModule
from fewshot.models.modules.nnlib import Linear
from fewshot.models.modules.container_module import ContainerModule
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModule('cpm_clean')
class CPM(ContainerModule):

  def __init__(self, name, proto_memory, rnn_memory, config, dtype=tf.float32):
    super(CPM, self).__init__(dtype=dtype)
    self._rnn_memory = rnn_memory
    self._proto_memory = proto_memory
    self._use_ssl = config.use_ssl  # CHECK
    D_in = self._rnn_memory.memory_dim
    D = self._rnn_memory.in_dim
    self._dim = D

    # h        [D]
    # scale    [D] deleted
    # gamma2   [1]
    # beta2    [1]
    # gamma    [1]
    # beta     [1]
    bias_init = [
        tf.zeros(D),
        # tf.zeros(D),
        tf.zeros([1]),
        tf.zeros([1]) + proto_memory.config.radius_init,
        tf.zeros([1]),
        tf.zeros([1]) + proto_memory.config.radius_init_write
    ]
    bias_init = tf.concat(bias_init, axis=0)
    D_out = bias_init.shape[-1]
    self._readout = Linear('readout', D_in, D_out, b_init=lambda: bias_init)

  @tf.function
  def forward(self, x, y, t, *states, ssl_store=None, **kwargs):
    D = self._dim
    rnn_states = states[:self._num_rnn_states]
    proto_states = states[self._num_rnn_states:]
    rnn_out, rnn_states_new = self._rnn_memory(x, *rnn_states)
    rnn_states = rnn_states_new
    readout = self.readout(rnn_out)
    h = readout[:, :D]
    # scale = tf.math.softplus(readout[:, D:2 * D])
    beta = readout[:, -1]
    gamma = tf.nn.softplus(readout[:, -2] + 1.0)
    beta2 = readout[:, -3]
    gamma2 = tf.nn.softplus(readout[:, -4] + 1.0)

    if not self._use_ssl:
      log.info('Disabling SSL compacity')
      ssl_store = tf.zeros([x.shape[0]], dtype=tf.bool)

    # Inference procedure.
    x = x + h
    pred_r = self.proto_memory.retrieve(
        x, t, *proto_states, beta=beta, gamma=gamma)
    pred_w = self.proto_memory.retrieve(
        x, t, *proto_states, beta=beta2, gamma=gamma2)

    # Memory store procedure.
    pred_w_unk = tf.math.sigmoid(pred_w[:, -1:])
    pred_w_cls = tf.nn.softmax(pred_w[:, :-1])
    y_soft = tf.concat([pred_w_cls * (1.0 - pred_w_unk), pred_w_unk], axis=-1)
    proto_states = self.proto_memory.store(
        x, y, *proto_states, y_soft=y_soft, ssl_store=ssl_store)
    D2 = proto_states[0].shape[-1]
    states = (*rnn_states, *proto_states)
    return pred_r, states

  def retrieve(self, x, t, *states, write=False, **kwargs):
    D = self._dim
    rnn_states = states[:self._num_rnn_states]
    proto_states = states[self._num_rnn_states:]
    rnn_out, rnn_states_new = self._rnn_memory(x, *rnn_states)
    rnn_states = rnn_states_new
    readout = self.readout(rnn_out)
    h = readout[:, :D]
    # scale = tf.math.softplus(readout[:, D:2 * D])
    beta = readout[:, -1]
    gamma = tf.nn.softplus(readout[:, -2] + 1.0)
    beta2 = readout[:, -3]
    gamma2 = tf.nn.softplus(readout[:, -4] + 1.0)

    if not self._use_ssl:
      log.info('Disabling SSL compacity')
      ssl_store = tf.zeros([x.shape[0]], dtype=tf.bool)

    # Inference procedure.
    x = x + h
    if write:
      return self.proto_memory.retrieve(
          x, t, *proto_states, beta=beta2, gamma=gamma2)
    else:
      return self.proto_memory.retrieve(
          x, t, *proto_states, beta=beta, gamma=gamma)

  def get_initial_state(self, bsize):
    rnn_init = self.rnn_memory.get_initial_state(bsize)
    proto_init = self.proto_memory.get_initial_state(bsize)
    self._num_rnn_states = len(rnn_init)
    self._num_proto_states = len(proto_init)
    return (*rnn_init, *proto_init)

  @property
  def readout(self):
    return self._readout

  @property
  def rnn_memory(self):
    return self._rnn_memory

  @property
  def proto_memory(self):
    return self._proto_memory
