""""Unit tests for proto memory."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest
import tensorflow as tf

from fewshot.models.modules.proto_memory_v2 import ProtoMemoryV2
from fewshot.configs.memory_net_config_pb2 import MemoryNetConfig


class ProtoMemoryTests(unittest.TestCase):

  def test_proto_memory(self):
    B = 2
    M = 3
    D = 2
    config = MemoryNetConfig()
    config.radius_init = 1.0
    config.radius_init_write = 1.0
    memory = ProtoMemoryV2("proto_memory", D, config)
    # states = memory.get_initial_state(B)
    storage = tf.constant([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                           [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
    usage = tf.constant([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    x = tf.constant([[0.8, 0.2], [2.5, 2.5]])
    y = tf.constant([4, 4], dtype=tf.int64)
    y_pred = memory.retrieve(x, 0, storage, usage)

    print('known', tf.nn.softmax(y_pred[:, :-1]))
    print('unknown', tf.nn.sigmoid(y_pred[:, -1]))
    states = (storage, usage)
    states = memory.store(x, y, *states, y_pred=y_pred, create_unk=True)
    print('storage', states[0])
    print('usage', states[1])
