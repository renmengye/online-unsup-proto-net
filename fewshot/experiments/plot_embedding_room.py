"""Plot nearest neighbors for Roaming Rooms.

Nearest across the same episode.
Nearest across different episodes?

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import os
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import split_eager_fallback
from tqdm import tqdm

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.utils import get_config
from fewshot.experiments.build_model import build_net
from fewshot.experiments.get_data_iter import get_dataiter_continual
from fewshot.experiments.get_data_iter import get_dataiter_sim
from fewshot.experiments.utils import latest_file
from fewshot.experiments.utils import get_data_fs
from fewshot.utils.logger import get as get_logger

log = get_logger()


def plot_nn(images, hidden, labels):
  """Plot the nearest neighbors"""
  hidden = tf.linalg.l2_normalize(hidden, axis=-1)
  sim = tf.matmul(hidden, hidden, transpose_b=True)
  idx = tf.argsort(sim, axis=-1, direction='DESCENDING')
  # N = images.shape[0]
  N = 100
  K = 10
  import cv2
  images = tf.transpose(images, [0, 2, 3, 1])
  for i in range(N):
    print(idx.shape, images.shape)
    top_k = tf.gather(images, idx[i, :K])  # [K, H, W, C]
    print(idx[i, :K])
    sim_k = tf.gather(sim[i], idx[i, :K])
    print('top k sim', sim_k, sim.shape, idx.shape, idx[i, :K].shape)
    labels_k = tf.gather(labels, idx[i, :K])
    print('inst label', labels[i], labels_k)
    recall = tf.reduce_sum(tf.cast(labels_k == labels[i], tf.int32))
    total = tf.reduce_sum(tf.cast(labels == labels[i], tf.int32))
    print('Recall @9', int(recall - 1), int(total - 1),
          float((recall - 1) / (total - 1)))
    if i == 61:
      assert False
    recallf = float((recall - 1) / (total - 1))
    H = top_k.shape[1]
    W = top_k.shape[2]
    C = top_k.shape[-1]
    top_k = tf.reshape(tf.transpose(top_k, [1, 0, 2, 3]),
                       [H, -1, C])  # [H, KxW, C]
    print(tf.reduce_max(top_k), tf.reduce_min(top_k), top_k.shape)
    alpha = top_k[:, :, -1:] * 0.3
    # alpha = alpha * 0.5 + 0.3
    red = tf.concat([tf.zeros([H, W * K, 2]), tf.ones([H, W * K, 1])], axis=-1)
    top_k = top_k[:, :, :3] + alpha * red  # + top_k[:, :, :3] * (1 - alpha)
    top_k = tf.clip_by_value(top_k, 0.0, 1.0)
    top_k = tf.cast(top_k * 255.0, tf.uint8)
    top_k = top_k.numpy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (255, 255, 255)
    thickness = 2
    for k in range(1, K):
      sim_ = sim_k[k].numpy()
      top_k = cv2.putText(top_k, '{:.2f}'.format(sim_), (W * k + 10, 20), font,
                          fontScale, color, thickness, cv2.LINE_AA)
    top_k = cv2.putText(top_k, 'Recall: {:.2f}'.format(recallf), (10, 20),
                        font, fontScale, color)
    cv2.imwrite('plots/debug-{}.png'.format(i), top_k)

    # Find out the missed items.
    if recall < total:
      import numpy as np
      total_idx = set(list(tf.where(labels == labels[i])[:, 0].numpy()))
      sel_idx = set(list(idx[i, :K].numpy()))
      diff = np.array(list(total_idx.difference(sel_idx)))
      diff_img = tf.gather(images, diff)
      diff_sim = tf.gather(sim[i], diff)
      K2 = len(diff)
      diff_img = tf.reshape(tf.transpose(diff_img, [1, 0, 2, 3]),
                            [H, -1, C])  # [H, KxW, C]
      red = tf.concat([tf.zeros([H, W * K2, 2]),
                       tf.ones([H, W * K2, 1])],
                      axis=-1)
      diff_img = diff_img[:, :, :3] + diff_img[:, :, -1:] * 0.3 * red
      diff_img = tf.clip_by_value(diff_img, 0.0, 1.0)
      diff_img = tf.cast(diff_img * 255.0, tf.uint8)
      diff_img = diff_img.numpy()
      for j in range(K2):
        sim_j = diff_sim[j].numpy()
        diff_img = cv2.putText(diff_img, '{:.2f}'.format(sim_j),
                               (W * j + 10, 20), font, fontScale, color,
                               thickness, cv2.LINE_AA)
      cv2.imwrite('plots/debug-{}-miss.png'.format(i), diff_img)


def run_images(model, dataiter, num_steps, verbose=False):
  """Run images through the network and get features.
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
  all_x = []
  all_h = []
  all_y = []
  for i, batch in zip(it, dataiter):
    x = batch['x_s']
    y = batch['y_abs']
    train_flag = batch['flag_s']
    T = tf.reduce_sum(train_flag)
    h = model.run_backbone(x, is_training=False)
    all_x.append(x[0, :T])  # [T, H, W, C]
    all_h.append(h[0, :T])  # [T, D]
    all_y.append(y[0, :T])  # [T]
  all_x = tf.concat(all_x, axis=0)
  all_h = tf.concat(all_h, axis=0)
  all_y = tf.concat(all_y, axis=0)
  print(all_x.shape, all_h.shape, all_y.shape)
  # assert False
  return all_x, all_h, all_y


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

  data_config.maxlen = config.num_steps
  data_config.unk_id = config.memory_net_config.max_classes
  log.info('Number of classes {}'.format(data_config.nway))
  log.info('Number of memory classes {}'.format(
      config.memory_net_config.max_classes))

  # Get dataset.
  dataset = get_data_fs(env_config, load_train=True)

  # Get data iterators.
  if env_config.dataset in ["matterport", "roaming-rooms"]:
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

  data['trainval_fs'].reset()
  data['val_fs'].reset()
  data['test_fs'].reset()

  split_list = ['test_fs']
  name_list = ['Test']
  # split_list = ['trainval_fs', 'val_fs', 'test_fs']
  # name_list = ['Train', 'Val', 'Test']
  nepisode_list = [args.nepisode, args.nepisode, args.nepisode]

  for split, name, N in zip(split_list, name_list, nepisode_list):
    data[split].reset()
    x, h, y = run_images(model, data[split], N)
    plot_nn(x, h, y)


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
