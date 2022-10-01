"""Plot cateogries learned

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.utils import get_config
from fewshot.experiments.build_model import build_net
from fewshot.experiments.get_data_iter import get_dataiter_continual
from fewshot.experiments.get_data_iter import get_dataiter_sim
from fewshot.experiments.get_stats import get_stats
from fewshot.experiments.get_stats import get_stats_unsup
from fewshot.experiments.utils import latest_file
from fewshot.experiments.utils import get_data_fs
from fewshot.utils.logger import get as get_logger

log = get_logger()


def select_threshold(model, data, nepisode, unsup=True):
  dist_q_list = np.linspace(0.0, 0.1, 25).astype(np.float32)
  best_score = 0.0
  best_dist_q = -1.0
  for dist_q in dist_q_list:
    r = evaluate(
        model,
        data,
        nepisode,
        verbose=False,
        reload=True,
        dist_q=tf.constant(dist_q))
    if unsup:
      stats = get_stats_unsup(r)
      print('dist', dist_q, 'MI', stats['mutual_info'])
      if stats['mutual_info'] > best_score:
        best_score = stats['mutual_info']
        best_dist_q = dist_q
      # print('best dist', best_dist_q, 'best MI', best_score)
    else:
      stats = get_stats(r)
      # print('dist', dist_q, 'AP', stats['ap'])
      if stats['ap'] > best_score:
        best_score = stats['ap']
        best_dist_q = dist_q

  return best_dist_q


def evaluate(model,
             dataiter,
             num_steps,
             verbose=False,
             reload=False,
             dist_q=None,
             plot=False):
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
    train_flag = batch['flag_s']
    y_gt = batch['y_gt']
    y_full = batch['y_full']
    kwargs = {'flag': train_flag, 'reload': reload}

    # Run model.
    if plot:
      pred = model.plot_step(
          x,
          y,
          batch['y_full'],
          'plot_unsup_{}_{}.png'.format(dataiter.dataset.split, i),
          dist_q=dist_q,
          **kwargs)
    else:
      pred = model.eval_step(x, y, dist_q=dist_q, **kwargs)

    # Support set metrics, accumulate per number of shots.
    y_np = y_full.numpy()  # [B, T]
    y_s_np = y.numpy()  # [B, T]
    y_gt_np = y_gt.numpy()  # [B, T]
    pred_np = pred.numpy()  # [B, T, K]
    pred_id_np = model.predict_id(pred).numpy()  # [B, T]

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

  print(data.keys())
  print(data)
  data['trainval_fs'].reset()
  data['val_fs'].reset()
  data['test_fs'].reset()

  split_list = ['trainval_fs', 'val_fs', 'test_fs']
  name_list = ['Train', 'Val', 'Test']
  nepisode_list = [args.nepisode, args.nepisode, args.nepisode]
  # nepisode_list = [args.nepisode]

  for split, name, N in zip(split_list, name_list, nepisode_list):
    data[split].reset()
    # Need to reload checkpoints in between to prevent contamination.
    # N = 1
    if data_config.unsupervised:
      dist_q = select_threshold(
          model, data[split], 10, unsup=data_config.unsupervised)
    else:
      dist_q = None
    # dist_q = 0.333
    evaluate(
        model,
        data[split],
        N,
        verbose=True,
        reload=True,
        dist_q=dist_q,
        plot=True)


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
