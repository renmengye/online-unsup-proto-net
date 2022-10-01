"""Synchronized batch normalization.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
from horovod.tensorflow.mpi_ops import _allreduce
from horovod.tensorflow.mpi_ops import size
from horovod.tensorflow.mpi_ops import Sum

from fewshot.models.modules.nnlib import BatchNorm


def reduce_sum(x):
  return _allreduce(x, op=Sum)


class SyncBatchNorm(BatchNorm):
  """Synchronized batch normalization layer."""

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""

    worker_mean, worker_variance = super(SyncBatchNorm, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    if size() > 1:
      # Compute variance using: Var[X] = E[X^2] - E[X]^2.
      worker_square_of_mean = tf.math.square(worker_mean)
      worker_mean_of_square = worker_variance + worker_square_of_mean

      # Average stats across all workers
      group_mean = reduce_sum(worker_mean)
      group_mean_of_square = reduce_sum(worker_mean_of_square)
      group_mean /= size()
      group_mean_of_square /= size()
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      # tf.print('mean', group_mean.shape, 'var', group_variance.shape)
      mean, var = group_mean, group_variance
    else:
      mean, var = worker_mean, worker_variance

    return mean, var
