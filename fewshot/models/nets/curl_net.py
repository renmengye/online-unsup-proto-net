from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from fewshot.models.nets.net import Net
from fewshot.models.nnlib import Linear
from fewshot.models.mlp import MLP
from fewshot.models.registry import RegisterModel
from fewshot.utils.logger import get as get_logger

from fewshot.models.modules.c4_backbone import ConvModule
from fewshot.models.modules.module import Module
from fewshot.models.modules.container_module import ContainerModule
from fewshot.models.variable_context import variable_scope

log = get_logger()


class BilinearModule(Module):

  def __init__(self, stride, data_format="NHWC"):
    self._stride = stride
    self._data_format = data_format

  def forward(self, x, **kwargs):

    if self._data_format == "NCHW":
      x = tf.transpose(x, [0, 2, 3, 1])

    H = x.shape[1]
    W = x.shape[2]
    s = self._stride
    x = tf.image.resize(
        x, [H * s, W * s], method=tf.image.ResizeMethod.BILINEAR)

    if self._data_format == "NCHW":
      x = tf.transpose(x, [0, 3, 1, 2])
    return x


class UpsampleConvNet(ContainerModule):
  """A conv net but stride is replaced by bilinear resizing."""

  def __init__(self, config, wdict=None):
    super(UpsampleConvNet, self).__init__(config)
    self._config = config
    assert len(config.pool) == 0
    assert not config.add_last_relu
    L = len(config.num_filters)
    if len(config.pool) > 0:
      pool = config.pool
    else:
      pool = [2] * L
    if config.normalization == "group_norm":
      num_groups = config.num_groups
    else:
      num_groups = [8] * L
    in_filters = [config.num_channels] + config.num_filters[:-1]
    add_relu = [True] * (L - 1) + [config.add_last_relu]
    self._module_list = []

    for i in range(len(config.num_filters)):
      if pool[i] > 1:
        self._module_list.append(
            BilinearModule(pool[i], data_format=config.data_format))
      self._module_list.append(
          ConvModule(
              "conv{}".format(i + 1),
              in_filters[i],
              config.num_filters[i],
              stride=1,
              add_relu=add_relu[i],
              pool_padding=config.pool_padding,
              data_format=config.data_format,
              normalization=config.normalization,
              num_groups=num_groups[i],
              wdict=wdict))

  def forward(self, x, is_training, **kwargs):
    for m in self._module_list:
      x = m(x, is_training=is_training)
    return x


@RegisterModel("curl_net")
class CURLNet(Net):

  def __init__(self,
               config,
               backbone,
               num_train_examples,
               distributed=False,
               dtype=tf.float32):
    super(CURLNet, self).__init__()
    self._backbone = backbone
    self._config = config
    self._dim = backbone.get_output_dimension()[-1]
    self._n_y = config.memory_net_config.max_items
    self._n_z = self._dim
    self._D = self._dim
    self._cluster_encoder = Linear(self._D, self._n_y)
    self._latent_encoder = Linear(self._D, self._n_y * 2 * n_z)
    self._decoder = self.build_decoder()

    # Something that needs to be manually set.
    self._n_y_active = self._get_variable(
        "n_y_active", lambda: tf.constant(n_y, dtype=tf.int64))

  def build_decoder(self):
    with variable_scope("decoder"):
      decoder = UpsampleConvNet(self._backbone.config)
    return decoder

  def run_cluster_encoder(self, hiddens, n_y_active, n_y):
    """
    Args:
      hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
      n_y_active: Tensor, the number of active components.
      n_y: int, number of maximum components allowed (used for tensor size).
    Returns:
      The distribution `q(y | x)`.
    """
    logits = self._cluster_encoder(hiddens)
    if n_y > 1:
      probs = tf.nn.softmax(logits[:, :n_y_active])
      paddings1 = tf.stack([tf.constant(0), tf.constant(0)], axis=0)
      paddings2 = tf.stack([tf.constant(0), n_y - n_y_active], axis=0)
      paddings = tf.stack([paddings1, paddings2], axis=1)
      probs = tf.pad(probs, paddings) + 0.0 * logits + 1e-12
    else:
      probs = tf.ones_like(logits)
    return tfp.distributions.OneHotCategorical(probs=probs)

  def generate_gaussian(self, logits, sigma_nonlin, sigma_param):
    """Generate a Gaussian distribution given a selected parameterisation."""

    mu, sigma = tf.split(value=logits, num_or_size_splits=2, axis=1)

    if sigma_nonlin == 'exp':
      sigma = tf.exp(sigma)
    elif sigma_nonlin == 'softplus':
      sigma = tf.nn.softplus(sigma)
    else:
      raise ValueError('Unknown sigma_nonlin {}'.format(sigma_nonlin))

    if sigma_param == 'var':
      sigma = tf.sqrt(sigma)
    elif sigma_param != 'std':
      raise ValueError('Unknown sigma_param {}'.format(sigma_param))

    return tfp.distributions.Normal(loc=mu, scale=sigma)

  def run_latent_encoder(self, hiddens, y):
    """The latent encoder function, modelling q(z | x, y).
    Args:
      hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
      y: Categorical cluster variable, `Tensor` of size `[B, n_y]`.
    Returns:
      The Gaussian distribution `q(z | x, y)`.
    """
    n_y = self.config.memory_net_config.max_items
    n_z = self._dim
    logits = tf.reshape(self._latent_encoder(hiddens), [-1, n_y, 2, n_z])
    y_ = y[:, :, None, None]  # [B, n_y, 1, 1]
    logits = tf.reduce_sum(logits * y, [1])  # [B, 2, n_z]
    return self.generate_gaussian(
        logits=logits, sigma_nonlin='softplus', sigma_param='var')

  def run_decoder(self, z, is_training=True):
    logits = self._decoder(z)
    out = tfp.distributions.Bernoulli(logits=logits)
    return out, logits

  def forward(self, x, is_training=True):
    hiddens = self._backbone(x, is_training=is_training)
    qy = self.run_cluster_encoder(hiddens, self._n_y_active, self._n_y)

    use_mode = True
    use_mean = False

    y_sample = qy.mode() if use_mode else qy.sample()
    y_sample = tf.to_float(y_sample)
    qz = self.run_latent_encoder(hiddens, y_sample)
    p = self.run_decoder(qz.sample())

    if use_mean:
      return p.mean()
    else:
      return p.sample()

  def log_prob_elbo(self, x):
    """Returns evidence lower bound."""
    log_p_x, kl_y, kl_z = self.log_prob_elbo_components(x)[:3]
    return log_p_x - kl_y - kl_z

  def log_prob_elbo_components(self, x, y=None, reduce_op=tf.reduce_sum):
    """Returns the components used in calculating the evidence lower bound.
    Args:
      x: Observed variables, `Tensor` of size `[B, I]` where `I` is the size of
        a flattened input.
      y: Optional labels, `Tensor` of size `[B, I]` where `I` is the size of a
        flattened input.
      reduce_op: The op to use for reducing across non-batch dimensions.
        Typically either `tf.reduce_sum` or `tf.reduce_mean`.
    Returns:
      `log p(x|y,z)` of shape `[B]` where `B` is the batch size.
      `KL[q(y|x) || p(y)]` of shape `[B]` where `B` is the batch size.
      `KL[q(z|x,y) || p(z|y)]` of shape `[B]` where `B` is the batch size.
    """
    cache_key = (x,)

    # Checks if the output graph for this inputs has already been computed.
    if cache_key in self._cache:
      return self._cache[cache_key]

    with tf.name_scope('{}_log_prob_elbo'.format(self.scope_name)):

      hiddens = self._shared_encoder(x, is_training=self._is_training)
      # 1) Compute KL[q(y|x) || p(y)] from x, and keep distribution q_y around
      kl_y, q_y = self._kl_and_qy(hiddens)  # [B], distribution

      # For the next two terms, we need to marginalise over all y.

      # First, construct every possible y indexing (as a one hot) and repeat it
      # for every element in the batch [n_y_active, B, n_y].
      # Note that the onehot have dimension of all y, while only the codes
      # corresponding to active components are instantiated
      bs, n_y = q_y.probs.shape
      all_y = tf.tile(
          tf.expand_dims(tf.one_hot(tf.range(self._n_y_active), n_y), axis=1),
          multiples=[1, bs, 1])

      # 2) Compute KL[q(z|x,y) || p(z|y)] (for all possible y), and keep z's
      # around [n_y, B] and [n_y, B, n_z]
      kl_z_all, z_all = tf.map_fn(
          fn=lambda y: self._kl_and_z(hiddens, y),
          elems=all_y,
          dtype=(tf.float32, tf.float32),
          name='elbo_components_z_map')
      kl_z_all = tf.transpose(kl_z_all, name='kl_z_all')

      # Now take the expectation over y (scale by q(y|x))
      y_logits = q_y.logits[:, :self._n_y_active]  # [B, n_y]
      y_probs = q_y.probs[:, :self._n_y_active]  # [B, n_y]
      y_probs = y_probs / tf.reduce_sum(y_probs, axis=1, keepdims=True)
      kl_z = tf.reduce_sum(y_probs * kl_z_all, axis=1)

      # 3) Evaluate logp and recon, i.e., log and mean of p(x|z,[y])
      # (conditioning on y only in the `multi` decoder_type case, when
      # train_supervised is True). Here we take the reconstruction from each
      # possible component y and take its log prob. [n_y, B, Ix, Iy, Iz]
      log_p_x_all = tf.map_fn(
          fn=lambda val: self.predict(val[0], val[1]).log_prob(x),
          elems=(z_all, all_y),
          dtype=tf.float32,
          name='elbo_components_logpx_map')

      # Sum log probs over all dimensions apart from the first two (n_y, B),
      # i.e., over I. Use einsum to construct higher order multiplication.
      log_p_x_all = snt.BatchFlatten(preserve_dims=2)(log_p_x_all)  # [n_y,B,I]
      # Note, this is E_{q(y|x)} [ log p(x | z, y)], i.e., we scale log_p_x_all
      # by q(y|x).
      log_p_x = tf.einsum('ij,jik->ik', y_probs, log_p_x_all)  # [B, I]

      # We may also use a supervised loss for some samples [B, n_y]
      if y is not None:
        self.y_label = tf.one_hot(y, n_y)
      else:
        self.y_label = tfc.placeholder(
            shape=[bs, n_y], dtype=tf.float32, name='y_label')

      # This is computing log p(x | z, y=true_y)], which is basically equivalent
      # to indexing into the correct element of `log_p_x_all`.
      log_p_x_sup = tf.einsum('ij,jik->ik', self.y_label[:, :self._n_y_active],
                              log_p_x_all)  # [B, I]
      kl_z_sup = tf.einsum('ij,ij->i', self.y_label[:, :self._n_y_active],
                           kl_z_all)  # [B]
      # -log q(y=y_true | x)
      kl_y_sup = tf.nn.sparse_softmax_cross_entropy_with_logits(  # [B]
          labels=tf.argmax(self.y_label[:, :self._n_y_active], axis=1),
          logits=y_logits)

      # Reduce over all dimension except batch.
      dims_x = [k for k in range(1, log_p_x.shape.ndims)]
      log_p_x = reduce_op(log_p_x, dims_x, name='log_p_x')
      log_p_x_sup = reduce_op(log_p_x_sup, dims_x, name='log_p_x_sup')

      # Store values needed externally
      self.q_y = q_y
      self.log_p_x_all = tf.transpose(
          reduce_op(
              log_p_x_all,
              -1,  # [B, n_y]
              name='log_p_x_all'))
      self.kl_z_all = kl_z_all
      self.y_probs = y_probs

    self._cache[cache_key] = (log_p_x, kl_y, kl_z, log_p_x_sup, kl_y_sup,
                              kl_z_sup)
    return log_p_x, kl_y, kl_z, log_p_x_sup, kl_y_sup, kl_z_sup

  def _kl_and_qy(self, hiddens):
    """Returns analytical or sampled KL div and the distribution q(y | x).
    Args:
      hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
    Returns:
      Pair `(kl, y)`, where `kl` is the KL divergence (a `Tensor` with shape
      `[B]`, where `B` is the batch size), and `y` is a sample from the
      categorical encoding distribution.
    """
    q = self.infer_cluster(hiddens)  # q(y|x)
    p = self.compute_prior()  # p(y)
    # Take the average proportions over whole batch then repeat it in each row
    # before computing the KL
    if self._kly_over_batch:
      probs = tf.reduce_mean(
          q.probs, axis=0, keepdims=True) * tf.ones_like(q.probs)
      qmean = tfp.distributions.OneHotCategorical(probs=probs)
      kl = tfp.distributions.kl_divergence(qmean, p)
    else:
      kl = tfp.distributions.kl_divergence(q, p)
    return kl, q

  def _kl_and_z(self, hiddens, y):
    """Returns KL[q(z|y,x) || p(z|y)] and a sample for z from q(z|y,x).
    Returns the analytical KL divergence KL[q(z|y,x) || p(z|y)] if one is
    available (as registered with `kullback_leibler.RegisterKL`), or a sampled
    KL divergence otherwise (in this case the returned sample is the one used
    for the KL divergence).
    Args:
      hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
      y: Categorical cluster random variable, `Tensor` of size `[B, n_y]`.
    Returns:
      Pair `(kl, z)`, where `kl` is the KL divergence (a `Tensor` with shape
      `[B]`, where `B` is the batch size), and `z` is a sample from the
      encoding distribution.
    """
    q = self.infer_latent(hiddens, y)  # q(z|x,y)
    p = self.generate_latent(y)  # p(z|y)
    z = q.sample(name='z')
    kl = tfp.distributions.kl_divergence(q, p)

    # Reduce over all dimension except batch.
    sum_axis_kl = [k for k in range(1, kl.get_shape().ndims)]
    kl = tf.reduce_sum(kl, sum_axis_kl, name='kl')
    return kl, z

  @tf.function
  def train_step(self, x):
    pass

  @property
  def backbone(self):
    return self._backbone

  @property
  def config(self):
    return self._config
