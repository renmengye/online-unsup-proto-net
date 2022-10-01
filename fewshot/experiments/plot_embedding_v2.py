"""Plot cateogries learned

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from warnings import filterwarnings

import numpy as np
import six
import sklearn.decomposition
import sklearn.manifold
import tensorflow as tf
import matplotlib
from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_net
from fewshot.experiments.get_data_iter import (get_dataiter_continual,
                                               get_dataiter_sim)
from fewshot.experiments.get_stats import get_stats, get_stats_unsup
from fewshot.experiments.metrics import convert_pred_to_full
from fewshot.experiments.utils import get_config, get_data_fs, latest_file
from fewshot.utils.logger import get as get_logger
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox  # NOQA
from tqdm import tqdm

log = get_logger()


def evaluate(model,
             dataiter,
             num_steps,
             verbose=False,
             reload=False,
             dist_q=None,
             threshold=0.5):
  """Evaluates online few-shot episodes.
  Args:
    model: Model instance.
    dataiter: Dataset iterator.
    num_steps: Number of episodes.
  """
  if num_steps == -1:
    it = six.moves.xrange(len(dataiter))
  else:
    it = six.moves.xrange(num_steps)
  if verbose:
    it = tqdm(it, ncols=0)
  results = []
  for i, batch in zip(it, dataiter):
    x = batch['x_s']
    y = batch['y_s']
    y = tf.zeros_like(y) + model.memory.max_classes
    y_gt = batch['y_gt']
    y_full = batch['y_full']
    train_flag = batch['flag_s']
    kwargs = {'flag': train_flag, 'reload': reload}

    # Run model.
    pred = model.eval_step(x, y, dist_q=dist_q, threshold=threshold, **kwargs)

    # Support set metrics, accumulate per number of shots.
    y_np = y_full.numpy()  # [B, T]
    y_s_np = y.numpy()  # [B, T]
    y_gt_np = y_gt.numpy()  # [B, T]
    pred_np = pred.numpy()  # [B, T, K]
    pred_id_np = model.predict_id(pred, threshold=threshold).numpy()  # [B, T]

    if train_flag is not None:
      flag_np = train_flag.numpy()
    else:
      flag_np = None

    results.append({
        'y_full': y_np,
        'y_gt': y_gt_np,
        'y_s': y_s_np,
        'pred': pred_np,
        'pred_id': pred_id_np,
        'flag': flag_np
    })
  return results


def select_threshold(model, data, nepisode):
  thresh_list = np.linspace(0.01, 0.99, 100).astype(np.float32)
  best_score = 0.0
  best_dist_q = -1.0
  for t in thresh_list:
    data.reset()
    r = evaluate(
        model,
        data,
        nepisode,
        verbose=False,
        reload=True,
        threshold=tf.constant(t))
    stats = get_stats_unsup(r)
    if stats['mutual_info'] > best_score:
      best_score = stats['mutual_info']
      best_dist_q = t
    print(t, stats['mutual_info'])
  return best_dist_q


def run_plot(model,
             dataiter,
             num_steps,
             verbose=False,
             reload=False,
             threshold=None,
             folder='.'):
  """Evaluates online few-shot episodes.
  Args:
    model: Model instance.
    dataiter: Dataset iterator.
    num_steps: Number of episodes.
  """
  # print(threshold)
  # assert False
  if num_steps == -1:
    it = six.moves.xrange(len(dataiter))
  else:
    it = six.moves.xrange(num_steps)
  if verbose:
    it = tqdm(it, ncols=0)
  results = []
  for i, batch in zip(it, dataiter):
    x = batch['x_s']
    y = tf.zeros_like(batch['y_s']) + 150
    # print(y, batch.keys())
    # assert False
    train_flag = batch['flag_s']
    y_gt = batch['y_gt']
    y_full = batch['y_full']
    kwargs = {'flag': train_flag, 'reload': reload}

    # Run model.
    model.memory.clear_state()
    # print(model.memory.)
    logits, _, states, _, _ = model.forward(
        x, y, is_training=False, threshold=threshold, **kwargs)
    pred_id = model.predict_id(logits, threshold=threshold).numpy()
    y_pred = convert_pred_to_full(pred_id)
    # print(y_pred)
    h = model.run_backbone(x)
    tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, random_state=1234)
    pca = sklearn.decomposition.PCA(n_components=2)
    # fig_w = 30
    # fig_h = 15
    fig_w = 20
    fig_h = 20
    plt.figure(figsize=(fig_w, fig_h))

    num_x = np.sum(kwargs['flag'])
    # print(x.shape, num_x)
    # assert False
    h = h[0, :num_x]
    y_pred = y_pred[0, :num_x]
    y_full = y_full[0, :num_x]
    num_k = np.max(y_pred) + 1

    all_hidden = h
    all_z = tsne.fit_transform(all_hidden)
    zmax = all_z.max(axis=0)
    zmin = all_z.min(axis=0)
    all_z = (all_z - zmin) / (zmax - zmin)

    # cmap = 'rainbow'
    # cmap = 'prism'
    # ax = plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.set_axis_off()
    # plt.axis('off')
    # plt.scatter(
    #     z[:, 0], z[:, 1], c=y_pred, cmap=cmap, alpha=0.3, s=30, linewidths=1)

    z = all_z[:num_x]
    k_list = np.arange(num_k)

    for j in range(num_x):
      points = z[j]
      img = x[0, j]
      if img.shape[-1] > 4:
        img = tf.transpose(img, [1, 2, 0])
        zoom = 0.25
        zoom2 = 0.5
        colorbox = True
      elif img.shape[-1] == 1:
        img = 1.0 - tf.tile(img, [1, 1, 3])
        zoom = 1.5
        zoom2 = 4.0
        colorbox = False
      else:
        zoom = 0.25
        zoom2 = 0.5
        colorbox = True
      if img.shape[-1] == 4:
        H = img.shape[0]
        W = img.shape[1]
        alpha = img[:, :, -1:] * 0.3
        red = tf.concat([tf.ones([H, W, 1]), tf.zeros([H, W, 2])], axis=-1)
        img = img[:, :, :3] + alpha * red
        img = tf.clip_by_value(img, 0.0, 1.0)

      # print(cmap[y_pred[j]])
      if colorbox:
        cmap_arr = matplotlib.cm.get_cmap('rainbow')
        color_idx = y_pred[j] / np.max(y_pred)
        rgba = cmap_arr(color_idx)
        imagebox = OffsetImage(
            img, zoom=zoom, interpolation='bicubic', zorder=-1)
        ab = AnnotationBbox(
            imagebox,
            tuple(points),
            pad=0.0,
            frameon=True,
            bboxprops=dict(edgecolor=rgba, linewidth=4.0))
      else:
        cmap_arr = matplotlib.cm.get_cmap('rainbow')
        color_idx = y_full[j] / np.max(y_full)
        rgba = cmap_arr(color_idx)
        mask = 1.0 - img[:, :, 0:1]
        rgba = np.array(rgba)
        neg = np.copy(rgba)
        neg[-1] = 0.0
        # print(mask, rgba)
        img = mask * rgba + (1.0 - mask) * neg
        imagebox = OffsetImage(
            img, zoom=zoom, interpolation='bicubic', zorder=-1)
        ab = AnnotationBbox(imagebox, tuple(points), pad=0.0, frameon=False)
      ax.add_artist(ab)

    for idx in k_list:
      points = z[y_pred == idx]
      centroid = np.mean(points, axis=0)

      # pick the image that is closest to the centroid
      dist = ((points - centroid)**2).sum()**0.5
      # print(dist)
      rep_idx = np.argmin(dist)
      # print(rep_idx)
      all_idx = np.nonzero(y_pred == idx)[0]
      # print('idx', idx)
      orig_idx = all_idx[rep_idx]
      rep_img = x[0, orig_idx]
      if rep_img.shape[-1] > 4:
        rep_img = tf.transpose(rep_img, [1, 2, 0])
      if rep_img.shape[-1] == 1:
        rep_img = 1.0 - np.tile(rep_img, [1, 1, 3])
      if rep_img.shape[-1] == 4:
        H = rep_img.shape[0]
        W = rep_img.shape[1]
        alpha = rep_img[:, :, -1:] * 0.3
        red = tf.concat([tf.ones([H, W, 1]), tf.zeros([H, W, 2])], axis=-1)
        rep_img = rep_img[:, :, :3] + alpha * red
        rep_img = tf.clip_by_value(rep_img, 0.0, 1.0)

      if colorbox:
        cmap_arr = matplotlib.cm.get_cmap('rainbow')
        color_idx = y_pred[orig_idx] / np.max(y_pred)
        rgba = cmap_arr(color_idx)
        imagebox = OffsetImage(rep_img, zoom=zoom2, interpolation='bicubic')
        ab = AnnotationBbox(
            imagebox,
            tuple(centroid),
            pad=0.0,
            frameon=True,
            bboxprops=dict(edgecolor=rgba, linewidth=4.0))
      else:
        cmap_arr = matplotlib.cm.get_cmap('rainbow')
        color_idx = y_full[orig_idx] / np.max(y_full)
        rgba = cmap_arr(color_idx)
        diameter = ((points.max(axis=0) - points.min(axis=0))**2).sum()**0.5
        circle = patches.Circle(
            tuple(centroid),
            diameter / 2 + 0.015,
            linewidth=4.0,
            facecolor='none',
            edgecolor='black',
            alpha=0.3,
            zorder=-10)
        ax.add_patch(circle)
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.12, 1.12)

        mask = 1.0 - rep_img[:, :, 0:1]
        # rgba = np.array(rgba)
        # neg = np.copy(rgba)
        # neg[-1] = 0.0
        rgba = np.array([0.0, 0.0, 0.0, 0.8])
        neg = np.zeros([4])
        rep_img = mask * rgba + (1.0 - mask) * neg
        imagebox = OffsetImage(
            rep_img,
            zoom=np.sqrt(points.shape[0]) * 1.8,
            interpolation='bicubic',
            zorder=-1)
        ab = AnnotationBbox(imagebox, tuple(centroid), pad=0.0, frameon=False)
      ax.add_artist(ab)
      ab.set_zorder(10)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'test-{}.png'.format(i)))


def main(args):
  log.info('Command line args {}'.format(args))
  data_config = get_config(args.data, EpisodeConfig)
  env_config = get_config(args.env, EnvironmentConfig)

  config = os.path.join(args.restore, 'config.prototxt')
  config = get_config(config, ExperimentConfig)
  # config.memory_net_config.similarity = "euclidean"
  model = build_net(config)
  model.load(latest_file(args.restore, 'weights-'))

  log.info('Model: \n{}'.format(config))
  log.info('Data episode: \n{}'.format(data_config))
  log.info('Environment: \n{}'.format(env_config))

  # data_config.maxlen = config.num_steps
  # assert False, str(config.num_steps)
  data_config.unk_id = config.memory_net_config.max_classes
  log.info('Number of classes {}'.format(data_config.nway))
  log.info('Number of memory classes {}'.format(
      config.memory_net_config.max_classes))

  # Get dataset.
  dataset = get_data_fs(env_config, load_train=True)

  # Get data iterators.
  if env_config.dataset in ["matterport", "roaming-rooms"]:
    print(data_config)
    data = get_dataiter_sim(
        dataset,
        data_config,
        batch_size=config.optimizer_config.batch_size,
        nchw=model.backbone.config.data_format == 'NCHW',
        seed=0)
  else:
    data = get_dataiter_continual(
        dataset,
        data_config,
        batch_size=config.optimizer_config.batch_size,
        nchw=model.backbone.config.data_format == 'NCHW',
        save_additional_info=True,
        random_box=data_config.random_box,
        seed=0)

  print(data.keys())
  print(data)
  data['trainval_fs'].reset()
  data['val_fs'].reset()
  data['test_fs'].reset()

  # split_list = ['trainval_fs', 'val_fs', 'test_fs']
  # name_list = ['Train', 'Val', 'Test']
  # split_list = ['test_fs']
  # name_list = ['Test']
  # split_list = ['trainval_fs']
  # name_list = ['Train']
  split_list = ['trainval_fs', 'test_fs']
  name_list = ['Train', 'Test']
  nepisode_list = [args.nepisode, args.nepisode, args.nepisode]
  # nepisode_list = [args.nepisode]
  for split, name, N in zip(split_list, name_list, nepisode_list):
    data[split].reset()
    # Need to reload checkpoints in between to prevent contamination.
    # N = 1
    output_folder = os.path.join(args.output, split)
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    # thresh = select_threshold(model, data[split], 10)
    # print(thresh)
    # assert False
    # thresh = 0.5940404 # RoamingRooms
    thresh = 0.49505052
    # thresh = 0.5
    run_plot(
        model,
        data[split],
        N,
        verbose=True,
        reload=True,
        threshold=thresh,
        folder=os.path.join(args.output, split))


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Lifelong Few-Shot Training')
  # Whether allow images to repeat in an episode.
  parser.add_argument('--bgr', action='store_true')
  parser.add_argument('--data', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--nepisode', type=int, default=10)
  parser.add_argument('--output', type=str, default='./output')
  parser.add_argument('--restore', type=str, default=None)
  args = parser.parse_args()
  main(args)
