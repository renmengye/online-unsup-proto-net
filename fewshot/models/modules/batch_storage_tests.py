""""Unit tests for blender."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest
import numpy as np
import tensorflow as tf

from fewshot.models.modules.batch_storage import BatchStorage


class BatchStorageTests(unittest.TestCase):

  def test_batch_storage(self):
    B = 2
    M = 10
    D = 4
    storage = BatchStorage(B, M, D, tf.float32)
    a1 = storage[tf.constant([3, 4])]
    print(a1)
    np.testing.assert_allclose(a1, np.zeros([B, D], dtype=np.float32))

    v = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=np.float32)
    storage[tf.constant([3, 4])] = tf.constant(v)
    a1 = storage[tf.constant([3, 4])]
    print(a1)
    np.testing.assert_allclose(a1, v)

    v2 = np.array([[5., 6., 3., 4.], [1., 2., 7., 8.]], dtype=np.float32)
    storage[tf.constant([3, 4])] = tf.constant(v2)
    a1 = storage[tf.constant([3, 4])]
    print(a1)
    np.testing.assert_allclose(a1, v2)


if __name__ == '__main__':
  unittest.main()
