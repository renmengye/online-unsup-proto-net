"""A recurrent net base class using sigmoid for getting a unknown output.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
import os
# import time
import uuid
import numpy as np

from fewshot.models.nets.episode_recurrent_net import EpisodeRecurrentNet
from fewshot.utils.dummy_context_mgr import dummy_context_mgr as dcm

from matplotlib import patches as patches
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox  # NOQA


class EpisodeRecurrentSigmoidNet(EpisodeRecurrentNet):
  """Episode recurrent network with sigmoid output."""

  def wt_avg(self, x, wt):
    """Weighted average.

    Args:
      x: [N]. Input.
      wt: [N]. Weight per input.
    """
    wsum = tf.reduce_sum(wt)
    delta = tf.cast(tf.equal(wsum, 0), self.dtype)
    wsum = tf.cast(wsum, self.dtype)
    wt = tf.cast(wt, self.dtype)
    return tf.reduce_sum(x * wt) / (wsum + delta)

  def xent_with_unk(self, logits, labels, K, flag):
    """Cross entropy with unknown sigmoid output with some flag."""
    # Gets unknowns.
    labels_unk = tf.cast(tf.equal(labels, K), self.dtype)  # [B, T+T']
    flag_unk = flag
    flag *= 1.0 - labels_unk

    # Do not compute loss if we predict that this is unknown.
    if self.config.disable_loss_self_unk:
      pred = self.predict_id(logits)
      is_logits_unk = tf.cast(tf.equal(pred, K), self.dtype)  # [B, T+T']
      flag *= 1.0 - is_logits_unk

    logits_unk = logits[:, :, -1]  # [B, T+T']
    logits = logits[:, :, :-1]  # [B, T+T', Kmax]
    labels = tf.math.minimum(
        tf.cast(tf.shape(logits)[-1], tf.int64) - 1,
        tf.cast(labels, tf.int64))  # [Kmax]

    # Cross entropy loss, either softmax or sigmoid.
    assert self.config.loss_fn == "softmax"
    if self.config.loss_fn == "softmax":
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)
    elif self.config.loss_fn == "sigmoid":
      labels_onehot = tf.one_hot(labels, K)
      xent = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=logits, labels=labels_onehot), [-1])

    # Binary cross entropy on unknowns.
    xent_unk = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_unk, labels=labels_unk)
    xent = self.wt_avg(xent, flag)
    xent_unk = self.wt_avg(xent_unk, flag_unk)
    # tf.print('xent', xent, 'xent_unk', xent_unk)
    return xent, xent_unk

  def xent(self, logits, labels, flag):
    """Cross entropy with some flag."""
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    xent = self.wt_avg(xent, flag)
    return xent

  def compute_loss(self, logits, y_gt, reg=True, flag=None, **kwargs):
    """Compute the training loss."""
    K = self.config.num_classes
    if flag is None:
      flag = tf.ones(tf.shape(y_gt), dtype=self.dtype)
    else:
      flag = tf.cast(flag, self.dtype)
    xent, xent_unk = self.xent_with_unk(logits, y_gt, K, flag=flag)

    # Regularizers.
    if reg:
      reg_loss = self._get_regularizer_loss(*self.regularized_weights())
      xent += self.config.unknown_loss * xent_unk
      loss = xent + reg_loss * self.wd
    else:
      loss = xent
    return loss, {
        # 'loss/all': loss,
        'loss/xent_cls': xent,
        'loss/xent_unk': xent_unk
    }

  # @tf.function
  def train_step(self, x, y, y_gt=None, flag=None, writer=None, **kwargs):
    """One training step.

    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep, to be fed as input.
      y_gt: [B, T], groundtruth at each timestep, if different from labels.
      flag: [B, T], if the current input is valid.

    Returns:
      xent: Cross entropy loss.
    """
    if self._distributed:
      import horovod.tensorflow as hvd

    with writer.as_default() if writer is not None else dcm() as gs:
      if self.config.optimizer_config.inner_loop_truncate_steps > 0:
        l, m, om = self._train_step_t(x, y, y_gt=y_gt, flag=flag, **kwargs)
        y_pred = None  # Not supported yet.
        y_pred1 = None
      else:
        y_pred1, y_pred, l, m, om = self._train_step(
            x, y, y_gt=y_gt, flag=flag, **kwargs)

      # Data parallel training, average loss.
      if self._distributed:
        for k in m:
          m[k] = tf.reduce_mean(
              hvd.allgather(tf.zeros([1], dtype=tf.float32) + m[k], name=k))

      # Writer add loss.
      write_flag = self._distributed and hvd.rank() == 0
      write_flag = write_flag or (not self._distributed)
      if write_flag and writer is not None:
        if tf.equal(
            tf.math.floormod(self._step + 1,
                             self.config.train_config.steps_per_log), 0):
          for k in m:
            tf.summary.scalar(k, m[k], step=self._step + 1)
          for k in om:
            tf.summary.scalar(k, om[k], step=self._step + 1)
          writer.flush()

        # # Extra visualization.
        if self.config.visualize_input:
          if tf.equal(tf.math.floormod(self._step - 1, 500), 0):
            x_ = x[0]
            if self.backbone.config.data_format == "NCHW":
              x_ = tf.transpose(x_, [0, 2, 3, 1])
            C_all = x_.shape[-1]
            C = self.backbone.config.num_channels
            M = C_all // C - 1
            inp = x_[:, :, :, :1 * C]
            if M >= 1:
              inp_a1 = x_[:, :, :, 1 * C:2 * C]
            else:
              inp_a1 = None
            if M >= 2:
              inp_a2 = x_[:, :, :, 2 * C:3 * C]
            else:
              inp_a2 = None

            MAX = 50

            def red_mask(x):
              mask = x[:, :, :, -1:]
              x = x[:, :, :, :-1]
              ones = tf.ones_like(x[:, :, :, :1])
              zeros = tf.zeros_like(x[:, :, :, :2])
              red = tf.concat([ones, zeros], axis=-1)
              red_mask = mask * red
              x = x * 0.5 + red_mask * 0.5
              x = tf.clip_by_value(x, 0.0, 1.0)
              return x

            def rgb2bgr(x):
              return x[:, :, :, [2, 1, 0]]

            def big_image(x, num_col):
              return np.reshape(
                  np.transpose(
                      np.reshape(
                          x,
                          [-1, num_col, x.shape[1], x.shape[2], x.shape[3]]),
                      [0, 2, 1, 3, 4]),
                  [1, -1, num_col * x.shape[2], x.shape[3]])

            if C == 3:
              inp = rgb2bgr(inp.numpy())
              if inp_a1 is not None:
                inp_a1 = rgb2bgr(inp_a1.numpy())
              if inp_a2 is not None:
                inp_a2 = rgb2bgr(inp_a2.numpy())
            elif C == 4:
              inp = red_mask(inp.numpy())
              if inp_a1 is not None:
                inp_a1 = red_mask(inp_a1.numpy())
              if inp_a2 is not None:
                inp_a2 = red_mask(inp_a2.numpy())

            # Draw prediction.
            if True:
              padding = 20
              inp_new = np.zeros([
                  inp.shape[0], inp.shape[1] + padding, inp.shape[2],
                  inp.shape[3]
              ])
              for i in range(inp.shape[0]):
                import cv2
                img = np.ascontiguousarray(np.copy(inp[i]))
                H, W, C = img.shape[0], img.shape[1], img.shape[2]
                img = np.concatenate([np.zeros([padding, W, C]), img], axis=0)
                img = cv2.putText(
                    img, 'idx: {:d} new: {:.1f}%'.format(
                        np.argmax(y_pred[0, i, :-1]),
                        1 / (1 + np.exp(-y_pred1[0, i, -1])) * 100.0),
                    (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                inp_new[i] = img

              big_inp_new = big_image(inp_new, 5)
              tf.summary.image('all', big_inp_new, step=self._step)
            # tf.summary.image('inp/x', inp_new, step=self._step,
            # max_outputs=MAX)
            if inp_a1 is not None:
              big_inp_a1 = big_image(inp_a1, 5)
              tf.summary.image('aug1', big_inp_a1, step=self._step)
              # tf.summary.image(
              #     'inp/a1', inp_a1, step=self._step, max_outputs=MAX)
            # if inp_a2 is not None:
            #   tf.summary.image(
            #       'inp/a2', inp_a2, step=self._step, max_outputs=MAX)

            # T_ = tf.cast(tf.reduce_min(tf.reduce_sum(flag, [1])),
            #              tf.int64)  # [B, T] => [B]
            # if self.backbone.config.data_format == "NCHW":
            #   x = x[:, :, :C, :, :]
            # else:
            #   x = x[:, :, :, :, :C]
            # x = self.run_backbone(x, is_training=True)
            # x_sel = x[:, :T_]
            # Tf = tf.cast(T_, tf.float32)
            # pdist = -self.memory.compute_logits_batch(x_sel, x_sel)  # [B, T, T]
            # pdist = tf.reshape(pdist, [-1])
            # pdist = tf.sort(pdist)
            # min_dist = pdist[0]
            # pdist = pdist[T_:]  # Remove the diagonals.
            # log_pdist = tf.math.log(pdist - min_dist)
            # pdist = pdist.numpy()
            # tf.summary.histogram('pdist', pdist, step=self._step)
            # tf.summary.histogram('log pdist', log_pdist, step=self._step)
            writer.flush()
    return l

  @tf.function
  def _train_step(self, x, y, y_gt=None, flag=None, **kwargs):
    """Standard full backprop."""
    if self._distributed:
      import horovod.tensorflow as hvd
    if y_gt is None:
      y_gt = y
    with tf.GradientTape() as tape:
      logits, logits2, _, o_loss, o_metric = self.forward(
          x, y, flag=flag, is_training=True, **kwargs)
      if self.config.optimizer_config.optimizer == 'lars':
        reg = False
      else:
        reg = True
      loss, metric = self.compute_loss(
          logits, y_gt, flag=flag, reg=reg, **kwargs)
      loss += o_loss

      # Apply gradients.
      if self._distributed:
        tape = hvd.DistributedGradientTape(tape)
    self.apply_gradients(loss, tape)
    # tf.print('applied')
    return logits, logits2, loss, metric, o_metric

  def _train_step_t(self, x, y, y_gt=None, flag=None, **kwargs):
    """Truncated backprop (full)."""
    if y_gt is None:
      y_gt = y
    B = tf.constant(x.shape[0])
    T = tf.constant(x.shape[1])
    states = self.memory.get_initial_state(B)
    DT = self.config.optimizer_config.inner_loop_truncate_steps
    metric_total = {}
    ometric_total = {}
    flag_total = tf.cast(tf.reduce_sum(flag), self.dtype)
    for t_start in range(0, self.config.num_steps, DT):
      t_end = tf.minimum(t_start + DT, T)
      x_ = x[:, t_start:t_end]
      y_ = y[:, t_start:t_end]
      lbl_ = y_gt[:, t_start:t_end]
      flg_ = flag[:, t_start:t_end]
      step_ = tf.equal(t_start, 0)
      f = self._train_step_tinner
      l, metric, states, o_metric = f(
          x_, y_, lbl_, flg_, *states, step=step_, **kwargs)
      flag_total_ = tf.reduce_sum(tf.cast(flg_, self.dtype))

      if tf.equal(t_start, 0):
        for k in metric:
          metric_total[k] = 0.0
        for k in o_metric:
          ometric_total[k] = 0.0

      for k in metric:
        metric_total[k] += metric[k] * flag_total_ / flag_total

      for k in o_metric:
        ometric_total[k] += o_metric[k] * flag_total_ / flag_total
    return l, metric_total, ometric_total

  @tf.function
  def _train_step_tinner(self, x, y, y_gt, flag, *states, step=True, **kwargs):
    """Truncated backprop (partial)."""
    if self._distributed:
      import horovod.tensorflow as hvd
    with tf.GradientTape() as tape:
      logits, states, o_loss, o_metric = self.forward(
          x, y, *states, is_training=True, flag=flag, **kwargs)
      loss, metric = self.compute_loss(logits, y_gt, flag, **kwargs)
      loss += o_loss
      # tf.print('loss', loss)
      # Apply gradients.
      if self._distributed:
        tape = hvd.DistributedGradientTape(tape)
      self.apply_gradients(loss, tape, add_step=step)
    return loss, metric, states, o_metric

  def _train_step_t2(self, x, y, y_gt=None, flag=None, **kwargs):
    """Overlap backprop at each step."""
    if y_gt is None:
      y_gt = y
    B = tf.constant(x.shape[0])
    T = tf.constant(x.shape[1])
    states = self.memory.get_initial_state(B)
    DT = self.config.optimizer_config.inner_loop_truncate_steps
    metric_total = {}
    flag_total = tf.cast(tf.reduce_sum(flag), self.dtype) * DT

    # Update every step.
    for t_start in tf.range(0, T):

      # Forward roll DT steps.
      t_end = tf.minimum(t_start + DT, T)
      x_ = x[:, t_start:t_end]
      y_ = y[:, t_start:t_end]
      lbl_ = y_gt[:, t_start:t_end]
      flg_ = flag[:, t_start:t_end]
      step_ = tf.equal(t_start, 0)
      loss, metric, states_ = self._train_step_tinner(
          x_, y_, lbl_, flg_, *states, step=step_, is_training=True)

      # Roll forward one step.
      _, states, _ = self.forward(
          x[:, t_start:t_start + 1],
          y[:, t_start:t_start + 1],
          *states,
          is_training=True)
      flag_total_ = tf.reduce_sum(tf.cast(flg_, self.dtype))

      if tf.equal(t_start, 0):
        for k in metric:
          metric_total[k] = 0.0

      for k in metric:
        metric_total[k] += metric[k] * flag_total_ / flag_total
    return loss, metric_total

  # @tf.function
  def eval_step(self, x, y, **kwargs):
    """One evaluation step.
    Args:
      x: [B, T, ...], inputs at each timestep.
      y: [B, T], label at each timestep.

    Returns:
      logits: [B, T, Kmax], prediction.
    """
    tsteps = self.config.optimizer_config.inner_loop_truncate_steps
    update = self.config.optimizer_config.inner_loop_update_eval
    if tsteps > 0 and update:
      return self._eval_step_t(x, y, **kwargs)
    else:
      return self._eval_step(x, y, **kwargs)

  @tf.function
  def _eval_step(self, x, y, **kwargs):
    """Standard eval."""
    logits, _, _, _, _ = self.forward(x, y, is_training=False, **kwargs)
    return logits

  # @tf.function
  def plot_step(self, x, y, y_full, fname, **kwargs):
    """Standard eval."""
    self.memory.clear_state()
    logits, _, states, _, _ = self.forward(x, y, is_training=False, **kwargs)
    pred_id = self.predict_id(logits).numpy()
    from fewshot.experiments.metrics import convert_pred_to_full
    y_pred = convert_pred_to_full(pred_id)
    from matplotlib import pyplot as plt
    import sklearn.decomposition
    import sklearn.manifold
    h = self.run_backbone(x)
    tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, random_state=1234)
    pca = sklearn.decomposition.PCA(n_components=2)
    fig = plt.figure(figsize=(10, 5))

    num_x = np.sum(kwargs['flag'])
    h = h[0, :num_x]
    y_pred = y_pred[0, :num_x]
    y_full = y_full[0, :num_x]
    num_k = np.max(y_pred) + 1

    # Both data points and cluster center.
    # Normalize the vectors.
    # h = tf.math.l2_normalize(h, axis=-1)
    # c = tf.math.l2_normalize(c, axis=-1)
    # c = states[0][0, :num_k]
    # all_hidden = tf.concat([h, c], axis=0)
    all_hidden = h

    all_z = tsne.fit_transform(all_hidden)
    zmax = all_z.max(axis=0)
    zmin = all_z.min(axis=0)
    all_z = all_z / (zmax - zmin)

    cmap = 'rainbow'
    # cmap = 'prism'
    ax = plt.subplot(1, 2, 1)
    ax.set_axis_off()
    # plt.scatter(
    #     z[:, 0], z[:, 1], c=y_pred, cmap=cmap, alpha=0.3, s=30, linewidths=1)

    z = all_z[:num_x]
    # k = all_z[num_x:]

    plt.scatter(
        z[:, 0], z[:, 1], c=y_full, cmap=cmap, alpha=0.3, s=30, linewidths=1)
    k_list = np.arange(num_k)

    for idx in k_list:
      points = z[y_pred == idx]
      centroid = np.mean(points, axis=0)

      # pick the image that is closest to the centroid
      dist = ((points - centroid)**2).sum()**0.5
      rep_idx = np.argmin(dist)
      all_idx = np.nonzero(y_pred == idx)[0]
      orig_idx = all_idx[rep_idx]
      rep_img = x[0, orig_idx]
      if rep_img.shape[-1] == 1:
        rep_img = 1.0 - np.tile(rep_img, [1, 1, 3])

      imagebox = OffsetImage(
          rep_img,
          zoom=np.sqrt(points.shape[0]) / 2.5,
          interpolation='bicubic',
          zorder=-1)
      ab = AnnotationBbox(imagebox, tuple(centroid), pad=0.0, frameon=False)
      ax.add_artist(ab)

      diameter = ((points.max(axis=0) - points.min(axis=0))**2).sum()**0.5
      circle = patches.Circle(
          tuple(centroid),
          diameter / 2 + 0.015,
          linewidth=1.5,
          facecolor='none',
          edgecolor='black',
          alpha=0.3)
      ax.add_patch(circle)
      ax.set_facecolor('none')
      ab.set_zorder(0)
      circle.set_zorder(2)

    # plt.scatter(
    #     k[:, 0], k[:, 1], c=k_list, cmap=cmap, alpha=0.3, s=220, linewidths=2)
    # for idx in k_list:
    #   plt.annotate('{:d}'.format(idx), (k[idx, 0], k[idx, 1]), fontsize=8)

    # Plot scores.
    import sklearn.metrics
    homo = sklearn.metrics.homogeneity_score(y_full, y_pred)
    comp = sklearn.metrics.completeness_score(y_full, y_pred)
    rand = sklearn.metrics.adjusted_rand_score(y_full, y_pred)
    ax.set_title('Rand={:.2f}, Homo={:.2f}, Comp={:.2f}'.format(
        rand, homo, comp))
    print('Rand={:.2f}, Homo={:.2f}, Comp={:.2f}'.format(rand, homo, comp))

    ax = plt.subplot(1, 2, 2)
    plt.scatter(
        z[:, 0], z[:, 1], c=y_full, cmap=cmap, alpha=1.0, s=50, linewidths=1)

    plt.title('Groundtruth')
    plt.savefig(fname)
    print(fname)
    return logits

  def _eval_step_t(self, x, y, **kwargs):
    """Eval with inner update (full)."""
    # Need to reload the checkpoint here, to prevent training the network on
    # test set.
    self._eval_optimizer = self._get_optimizer(
        self.config.optimizer_config.optimizer, 1e-4)
    if 'reload' in kwargs and kwargs['reload'] is True:
      tmp_path = str(uuid.uuid4()) + '.tmp'
      self.save(tmp_path)
    flag = kwargs['flag']  # Need to pass in additional parameter here.
    del kwargs['flag']
    flag = tf.cast(flag, tf.float32)

    # Recreate training for unk flag.
    K = self.config.num_classes
    y_np = y.numpy()
    cummax = np.maximum.accumulate(y_np, axis=1)
    cond = y_np[:, 1:] > cummax[:, :-1]
    y_np[:, 0] = K
    y_np[:, 1:] = np.where(cond, K, y_np[:, 1:])

    # Create y_gt, Assuming we are not in semisupervised mode.
    y_gt = tf.constant(y_np)

    # # Do not update when the label is `UNK`
    # flag = flag * tf.where(tf.equal(y, K), 0.0, 1.0)
    # y_gt = y

    B = tf.constant(x.shape[0])
    T = tf.constant(x.shape[1])
    states = self.memory.get_initial_state(B)
    DT = self.config.optimizer_config.inner_loop_truncate_steps
    num_steps = int(np.ceil(T / DT))
    logits_all = tf.TensorArray(self.dtype, size=num_steps)
    for i in tf.range(num_steps):
      t_start = i * DT
      t_end = tf.minimum(t_start + DT, T)
      # print('states', t_start, states[0].shape, states[1].shape, len(states))
      x_ = x[:, t_start:t_end]
      y_ = y[:, t_start:t_end]
      lbl_ = y_gt[:, t_start:t_end]
      flg_ = flag[:, t_start:t_end]
      f = self._eval_step_tinner
      # [1, DT, K]
      logits, states = f(x_, y_, lbl_, flg_, *states, **kwargs)
      logits_all = logits_all.write(i, logits)
    logits_all = logits_all.stack()  # [NT, 1, DT, K]
    logits_all = tf.reshape(tf.transpose(logits_all, [0, 2, 1, 3]), [B, T, -1])
    if 'reload' in kwargs and kwargs['reload'] is True:
      self.load(tmp_path, verbose=False)
      os.remove(tmp_path)
    return logits_all

  def _eval_step_t2(self, x, y, **kwargs):
    """Eval with inner update (full)."""
    # Need to reload the checkpoint here, to prevent training the network on
    # test set.
    self._eval_optimizer = self._get_optimizer(
        self.config.optimizer_config.optimizer, 1e-4)
    if 'reload' in kwargs and kwargs['reload'] is True:
      tmp_path = str(uuid.uuid4()) + '.tmp'
      self.save(tmp_path)
    flag = kwargs['flag']  # Need to pass in additional parameter here.
    del kwargs['flag']
    flag = tf.cast(flag, tf.float32)

    # Recreate training for unk flag.
    K = self.config.num_classes
    y_np = y.numpy()
    cummax = np.maximum.accumulate(y_np, axis=1)
    cond = y_np[:, 1:] > cummax[:, :-1]
    y_np[:, 0] = K
    y_np[:, 1:] = np.where(cond, K, y_np[:, 1:])

    # Create y_gt, Assuming we are not in semisupervised mode.
    y_gt = tf.constant(y_np)

    B = tf.constant(x.shape[0])
    T = tf.constant(x.shape[1])
    states = self.memory.get_initial_state(B)
    DT = self.config.optimizer_config.inner_loop_truncate_steps
    logits_all = tf.TensorArray(self.dtype, size=T)
    states_all = [None] * int(T + 1)
    states_all[0] = states

    for i in range(T):
      # Regular forward pass.
      x_ = x[:, i:i + 1]
      y_ = y[:, i:i + 1]
      logits, states, _ = self.forward(x_, y_, *states, is_training=False)
      states_all[i + 1] = states
      logits_all = logits_all.write(i, logits)

      # Rehearsal on the past time window.
      if i >= DT:
        i_ = tf.maximum(0, i - DT + 1)
        x_ = x[:, i_:i + 1]
        y_ = y[:, i_:i + 1]
        lbl_ = y_gt[:, i_:i + 1]
        flg_ = flag[:, i_:i + 1]
        f = self._eval_step_tinner
        _, _ = f(x_, y_, lbl_, flg_, *states_all[i_], **kwargs)

    logits_all = logits_all.stack()  # [NT, 1, DT, K]
    logits_all = tf.reshape(tf.transpose(logits_all, [0, 2, 1, 3]), [B, T, -1])

    if 'reload' in kwargs and kwargs['reload'] is True:
      self.load(tmp_path, verbose=False)
      os.remove(tmp_path)
    return logits_all

  def _eval_step_tinner(self, x, y, y_gt, flag, *states, **kwargs):
    """Eval with inner update (partial)."""
    B = x.shape[0]
    T = x.shape[1]
    T2 = self.config.optimizer_config.inner_loop_repeat
    K = self.config.num_classes
    logits_out = tf.zeros([B, T, K + 1])
    states_out = states
    assert not self.config.memory_net_config.use_variables
    assert T2 == 0
    for i in tf.range(T2 + 1):
      with tf.GradientTape() as tape:
        logits_, states_, _ = self.forward(x, y, *states, is_training=False)
        if i < T2 + 1:
          loss, metric = self.compute_loss(logits_, y_gt, flag, **kwargs)
          self.apply_gradients(
              loss, tape, opt=self._eval_optimizer, add_step=False)
        tf.print(i, 'loss', loss, 'y gt', y_gt)
        if tf.equal(i, 0):
          logits_out = logits_
          states_out = states_
    return logits_out, states_out

  def predict_id(self, logits, threshold=0.5):
    """Predict class ID based on logits."""
    # print('logits', logits[:, :, -1])
    unk = tf.greater(tf.math.sigmoid(logits[:, :, -1]), threshold)  # [B, T]
    # print('unk', unk)
    # print(self.config.num_classes)
    # assert False
    non_unk = tf.cast(
        tf.argmax(logits[:, :, :-1], axis=-1), dtype=tf.int32)  # [B, T]
    final = tf.where(unk, self.config.num_classes, non_unk)
    return final
