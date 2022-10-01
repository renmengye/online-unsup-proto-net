""""Unit tests for LRU storage."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest
import numpy as np
import tensorflow as tf

from fewshot.models.modules.batch_lru_storage import BatchLRUStorage


class BatchLRUStorageTests(unittest.TestCase):

  def test_batch_lru_storage(self):
    B = 2
    M = 3
    D = 1
    storage = BatchLRUStorage(B, M, D, decay=0.98, dtype=tf.float32)
    a1 = storage[tf.constant([0, 1])]
    print(a1)
    key = storage.allocate_new_key()
    print(key)
    storage.write_sparse(key, tf.constant([[1.], [2.]]))
    print(storage.all)
    np.testing.assert_allclose(storage.all.numpy().reshape([-1]),
                               np.array([[1, 0, 0], [2, 0, 0]]).reshape([-1]))
    print(storage._usage.all)
    key = storage.allocate_new_key()
    storage.write_sparse(key, tf.constant([[3.], [4.]]))
    print(storage.all)
    np.testing.assert_allclose(storage.all.numpy().reshape([-1]),
                               np.array([[1, 3, 0], [2, 4, 0]]).reshape([-1]))
    print(storage._usage.all)
    key = storage.allocate_new_key()
    storage.write_sparse(key, tf.constant([[5.], [6.]]))
    print(storage.all)
    np.testing.assert_allclose(storage.all.numpy().reshape([-1]),
                               np.array([[1, 3, 5], [2, 4, 6]]).reshape([-1]))
    print(storage._usage.all)
    key = storage.allocate_new_key()
    storage.write_sparse(key, tf.constant([[7.], [8.]]))
    print(storage.all)
    np.testing.assert_allclose(storage.all.numpy().reshape([-1]),
                               np.array([[7, 3, 5], [8, 4, 6]]).reshape([-1]))
    print(storage._usage.all)
    key = storage.allocate_new_key(mask=tf.constant([0.0, 1.0]))
    print(storage.all)
    np.testing.assert_allclose(storage.all.numpy().reshape([-1]),
                               np.array([[7, 3, 5], [8, 0, 6]]).reshape([-1]))
    print(storage._usage.all)
    storage.write_dense(
        tf.constant([[0.3, 0.7, 0.0], [0.0, 0.0, 1.0]]),
        tf.constant([[7.], [8.]]))
    print(storage.all)
    np.testing.assert_allclose(
        storage.all.numpy().reshape([-1]),
        np.array([[7.0, 4.706077, 5], [8, 0, 7.0202]]).reshape([-1]))
    print(storage._usage.all)

  def test_key_alloc(self):
    B = 2
    M = 3
    D = 1
    storage = BatchLRUStorage(B, M, D, decay=0.99, dtype=tf.float32)
    for i in range(10):
      key = storage.allocate_new_key()
      storage.write_sparse(key, tf.constant([[i], [i]], dtype=tf.float32))
      print(i, key)


if __name__ == '__main__':
  unittest.main()
