"""Synchronized batch norm using horovod and keras"""
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers.normalization import BatchNormalizationBase
from tensorflow.python.util.tf_export import keras_export
from horovod.tensorflow.mpi_ops import _allreduce
from horovod.tensorflow.mpi_ops import size
from horovod.tensorflow.mpi_ops import Sum


def reduce_sum(x):
  return _allreduce(x, op=Sum)


# pylint: disable=g-classes-have-attributes
@keras_export('keras.layers.experimental.SyncBatchNormalizationHorovod', v1=[])
class SyncBatchNormalizationHorovod(BatchNormalizationBase):
  r"""Normalize and scale inputs or activations synchronously across replicas.
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               **kwargs):
    if kwargs.pop('fused', None):
      raise ValueError(
          '`fused` argument cannot be True for SyncBatchNormalization.')

    # Currently we only support aggregating over the global batch size.
    super(SyncBatchNormalizationHorovod, self).__init__(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        fused=False,
        **kwargs)

  def _calculate_mean_and_var(self, x, axes, keep_dims):

    with backend.name_scope('moments'):
      y = tf.cast(x, tf.float32) if x.dtype == tf.float16 else x

      # Compute true mean while keeping the dims for proper broadcasting.
      worker_mean = tf.reduce_mean(y, axes, keepdims=True, name='mean')
      worker_variance = tf.reduce_mean(
          tf.math.squared_difference(y, tf.stop_gradient(worker_mean)),
          axes,
          keepdims=True,
          name='variance')
      if size() > 1:
        worker_square_of_mean = tf.math.square(worker_mean)
        worker_mean_of_square = worker_variance + worker_square_of_mean
        group_mean = reduce_sum(worker_mean)
        group_mean_of_square = reduce_sum(worker_mean_of_square)
        group_mean /= size()
        group_mean_of_square /= size()
        group_variance = group_mean_of_square - tf.math.square(group_mean)
        mean, var = group_mean, group_variance
      else:
        mean, var = worker_mean, worker_variance

      if not keep_dims:
        mean = tf.squeeze(mean, axes)
        var = tf.squeeze(var, axes)
      if x.dtype == tf.float16:
        return (tf.cast(mean, tf.float16), tf.cast(var, tf.float16))
      else:
        return (mean, var)
