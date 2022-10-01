"""
Train an online few-shot network.
Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle as pkl

import numpy as np
import six
import tensorflow as tf
import horovod.tensorflow as hvd
tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
hvd.init()
if gpus:
  tf.config.experimental.set_visible_devices(
      gpus[hvd.local_rank() % len(gpus)], 'GPU')
is_chief = hvd.rank() == 0

# if __name__ == '__main__':
#   tf.get_logger().setLevel('ERROR')
#   gpus = tf.config.experimental.list_physical_devices('GPU')
#   if len(gpus) > 1:
#     import horovod.tensorflow as hvd
#     hvd.init()
#     if gpus:
#       tf.config.experimental.set_visible_devices(
#           gpus[hvd.local_rank() % len(gpus)], 'GPU')
#     is_chief = hvd.rank() == 0
#   else:
#     is_chief = True
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import gc

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_net
from fewshot.experiments.get_data_iter import (
    get_dataiter_continual, get_dataiter_sim, get_dataiter_vid)
from fewshot.experiments.get_stats import (get_stats, get_stats_unsup,
                                           log_results, log_results_unsup)
from fewshot.experiments.utils import (ExperimentLogger, delete_old_ckpt,
                                       get_config, get_data_fs, latest_file,
                                       save_config)
from fewshot.utils.dummy_context_mgr import dummy_context_mgr as dcm
from fewshot.utils.logger import get as get_logger
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
    # if unsup:
    #   y = tf.zeros_like(batch['y_s']) + model.memory.max_classes
    y_gt = batch['y_gt']
    y_full = batch['y_full']
    train_flag = batch['flag_s']
    kwargs = {'flag': train_flag, 'reload': reload}

    # Run model.
    pred = model.eval_step(x, y, dist_q=dist_q, threshold=threshold, **kwargs)
    # pred = model.plot_step(
    #     x, y, batch['y_full'], 'plot_unsup_{}_{}.png'.format(
    #         dataiter.dataset.split, i), **kwargs)

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


def train(model,
          dataiter,
          dataiter_traintest,
          dataiter_test,
          ckpt_folder,
          final_save_folder=None,
          nshot_max=5,
          maxlen=40,
          logger=None,
          writer=None,
          is_chief=True,
          reload_flag=None,
          unsup=False,
          save_all=False):
  """Trains the online few-shot model.
  Args:
    model: Model instance.
    dataiter: Dataset iterator.
    dataiter_test: Dataset iterator for validation.
    save_folder: Path to save the checkpoints.
  """
  N = model.max_train_steps
  config = model.config.train_config

  def try_log(*args, **kwargs):
    if logger is not None:
      logger.log(*args, **kwargs)

  def try_flush(*args, **kwargs):
    if logger is not None:
      logger.flush()

  r = None
  best_val = 0.0
  start = model.step.numpy()
  if start > 0:
    log.info('Restore from step {}'.format(start))

  it = six.moves.xrange(start, N)
  if is_chief:
    it = tqdm(it, ncols=0)
  import horovod.tensorflow as hvd
  distributed = hvd.size() > 1
  for i, batch in zip(it, dataiter):
    if distributed:
      print(hvd.allgather(tf.constant([i])))
      print('rank', hvd.rank(), 'data id', batch['id'])
    tf.summary.experimental.set_step(i + 1)
    x = batch['x_s']
    y = batch['y_s']
    y = batch['y_s']
    y_gt = batch['y_gt']
    if args.dist_q > 0.0:
      dist_q = tf.constant(args.dist_q)
      # Warm up with 3000.
      # dist_q = tf.constant(
      #     np.minimum(3000, i) / 3000.0 * args.dist_q, dtype=tf.float32)
    else:
      dist_q = None

    kwargs = {'y_gt': y_gt, 'flag': batch['flag_s'], 'dist_q': dist_q}
    kwargs['writer'] = writer

    loss = model.train_step(x, y, **kwargs)

    if i == start and reload_flag is not None:
      print(reload_flag)
      model.load(reload_flag, load_optimizer=True)

    # Synchronize distributed weights.
    if i == start and model._distributed:
      import horovod.tensorflow as hvd
      hvd.broadcast_variables(model.var_to_optimize(), root_rank=0)
      hvd.broadcast_variables(model.optimizer.variables(), root_rank=0)
      if model.config.set_backbone_lr:
        hvd.broadcast_variables(model._bb_optimizer.variables(), root_rank=0)

    # Evaluate.
    # if is_chief and ((i + 1) % config.steps_per_val == 0 or i == 0):
    if is_chief and ((i + 1) % config.steps_per_val == 0):
      for key, data_it_ in zip(['train', 'val'],
                               [dataiter_traintest, dataiter_test]):
        data_it_.reset()
        reload_ = model.config.optimizer_config.inner_loop_update_eval
        if unsup:
          # Threshold not selected
          r1 = evaluate(model, data_it_, 60, reload=reload_)
          r = get_stats_unsup(r1)
          try_log('unsup/mutual_info {}'.format(key), i + 1,
                  r['mutual_info'] * 100.0)
          try_log('unsup/rand {}'.format(key), i + 1, r['rand'] * 100.0)
          try_log('unsup/homogeneity {}'.format(key), i + 1,
                  r['homogeneity'] * 100.0)
          try_log('unsup/completeness {}'.format(key), i + 1,
                  r['completeness'] * 100.0)
          if args.select_threshold:
            if args.thresh_compat:
              dist_q = select_threshold_compat(model, data_it_, 10)
              thresh = 0.5
            else:
              thresh = select_threshold(model, data_it_, 10)
              dist_q = None
            r1 = evaluate(
                model,
                data_it_,
                60,
                dist_q=dist_q,
                threshold=thresh,
                reload=reload_)
            r = get_stats_unsup(r1)
            try_log('unsup/mutual_info max {}'.format(key), i + 1,
                    r['mutual_info'] * 100.0)
            try_log('unsup/rand max {}'.format(key), i + 1, r['rand'] * 100.0)
            try_log('unsup/homogeneity max {}'.format(key), i + 1,
                    r['homogeneity'] * 100.0)
            try_log('unsup/completeness max {}'.format(key), i + 1,
                    r['completeness'] * 100.0)
        else:
          r1 = evaluate(model, data_it_, 60, reload=reload_)
          r = get_stats(r1, nshot_max=nshot_max, tmax=maxlen)
          for s in range(nshot_max):
            try_log('online fs acc/{} s{}'.format(key, s), i + 1,
                    r['acc_nshot'][s] * 100.0)
          try_log('online fs ap/{}'.format(key), i + 1, r['ap'] * 100.0)
      try_log('lr', i + 1, model.learn_rate())
      print()

    # Save.
    if is_chief and ((i + 1) % config.steps_per_save == 0):
      model.save(os.path.join(ckpt_folder, 'weights-{}'.format(i + 1)))
      model.save(os.path.join(final_save_folder, 'weights-{}'.format(i + 1)))
      try_flush()

      # Delete earlier checkpoint.
      if not save_all:
        delete_old_ckpt(final_save_folder, "weights-", i + 1,
                        1 * config.steps_per_save)
      # Save the best checkpoint.
      if r is not None:
        if unsup:
          val_metric = r['mutual_info']
        else:
          val_metric = r['ap']
        if val_metric > best_val:
          model.save(os.path.join(final_save_folder, 'best-{}'.format(i + 1)))
          best_val = val_metric

          # Delete earlier best checkpoint.
          if not save_all:
            delete_old_ckpt(final_save_folder, "best-", i + 1, 0)

      # Memory leak hack.
      gc.collect()

    # Write logs.
    if is_chief and ((i + 1) % config.steps_per_log == 0 or i == 0):
      try_log('loss/all', i + 1, loss)

      # Update progress bar.
      post_fix_dict = {}
      post_fix_dict['lr'] = '{:.3e}'.format(model.learn_rate())
      post_fix_dict['loss'] = '{:.3e}'.format(loss)
      if r is not None:
        if unsup:
          post_fix_dict['mi_val'] = '{:.3f}'.format(r['mutual_info'] * 100.0)
        else:
          post_fix_dict['ap_val'] = '{:.3f}'.format(r['ap'] * 100.0)
      it.set_postfix(**post_fix_dict)

  # Save.
  if is_chief and final_save_folder is not None:
    model.save(os.path.join(final_save_folder, 'weights-{}'.format(N)))


def select_threshold_compat(model, data, nepisode):
  if args.large_sweep:
    dist_q_list = np.linspace(0.0, 0.3, 25).astype(np.float32)
  else:
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
    stats = get_stats_unsup(r)
    print('dist_q', dist_q, 'MI', stats['mutual_info'])
    if stats['mutual_info'] > best_score:
      best_score = stats['mutual_info']
      best_dist_q = dist_q
  return best_dist_q


def select_threshold(model, data, nepisode):
  # thresh_list = np.linspace(0.3, 0.7, 30).astype(np.float32)
  # thresh_list = np.linspace(0.1, 0.9, 100).astype(np.float32)
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
    print('thresh', t, 'MI', stats['mutual_info'])
    if stats['mutual_info'] > best_score:
      best_score = stats['mutual_info']
      best_dist_q = t
  return best_dist_q


def main():
  assert tf.executing_eagerly(), 'Only eager mode is supported.'
  assert args.config is not None, 'You need to pass in model config file path'
  assert args.data is not None, 'You need to pass in episode config file path'
  assert args.env is not None, 'You need to pass in environ config file path'
  assert args.tag is not None, 'You need to specify a tag'

  log.info('Command line args {}'.format(args))
  config = get_config(args.config, ExperimentConfig)
  data_config = get_config(args.data, EpisodeConfig)
  env_config = get_config(args.env, EnvironmentConfig)
  log.info('Model: \n{}'.format(config))
  log.info('Data episode: \n{}'.format(data_config))
  log.info('Environment: \n{}'.format(env_config))
  # config.num_classes = data_config.maxlen  # TODO: change this.

  if args.maxlen > 0:
    log.info('New max len modified to {:.3f}'.format(args.maxlen))
    data_config.maxlen = args.maxlen

  config.num_steps = data_config.maxlen

  if args.max_classes > 0:
    # Manual case.
    config.num_classes = args.max_classes  # Assign num classes.
    config.memory_net_config.max_classes = args.max_classes
    data_config.unk_id = args.max_classes
    data_config.nway = args.max_classes
  elif config.memory_net_config.create_unk:
    # Unsupervised case.
    config.num_classes = data_config.maxlen  # Assign num classes.
    config.memory_net_config.max_classes = data_config.maxlen
    data_config.unk_id = data_config.maxlen
  else:
    # Standard supervised case.
    config.num_classes = data_config.nway  # Assign num classes.
    config.memory_net_config.max_classes = data_config.nway
    data_config.unk_id = data_config.nway
  config.memory_net_config.max_stages = data_config.nstage
  config.memory_net_config.max_items = data_config.maxlen
  config.oml_config.num_classes = data_config.nway
  config.fix_unknown = data_config.fix_unknown  # Assign fix unknown ID.
  log.info('Number of classes {}'.format(data_config.nway))
  log.info('Number of memory classes {}'.format(
      config.memory_net_config.max_classes))

  # By pass config.
  # print(config.memory_net_config.new_prob_target)
  # assert False
  if args.new_prob_target > 0.0:
    log.info('New target prob modified to {:.3f}'.format(args.new_prob_target))
    config.memory_net_config.new_prob_target = args.new_prob_target

  if args.new_cluster_thresh > 0.0:
    log.info('New cluster thresh modified to {:.3f}'.format(
        args.new_cluster_thresh))
    config.memory_net_config.new_cluster_thresh = args.new_cluster_thresh

  if args.area_lb > 0.0:
    log.info('Area LB modified to {:.3f}'.format(args.area_lb))
    data_config.area_lb = args.area_lb

  if args.num_aug > 0:
    log.info('Num aug modified to {}'.format(args.num_aug))
    data_config.num_aug = args.num_aug

  if args.imbalance_ratio > 0.0:
    log.info('Imbalance ratio modified to {}'.format(args.imbalance_ratio))
    data_config.imbalance = True
    data_config.imbalance_ratio = args.imbalance_ratio

  if args.frame_rate > 0:
    log.info('Frame rate modified to {}'.format(args.frame_rate))
    data_config.frame_rate = args.frame_rate

  if args.random_rotate >= 0.0:
    log.info('Random rotate modified to {:.3f}'.format(args.random_rotate))
    data_config.random_rotate = True
    data_config.random_rotate_angle = args.random_rotate

  if args.steps_per_val > 0:
    log.info('Steps per val modified to {}'.format(args.steps_per_val))
    config.train_config.steps_per_val = args.steps_per_val

  if args.entropy_loss >= 0.0:
    log.info('Entropy loss modified to {:.3f}'.format(args.entropy_loss))
    config.memory_net_config.entropy_loss = args.entropy_loss

  if args.soft_label_temp >= 0.0:
    log.info('Soft label temp modified to {:.3f}'.format(args.soft_label_temp))
    config.memory_net_config.soft_label_temp = args.soft_label_temp

  if args.new_entropy_loss >= 0.0:
    log.info('New entropy loss modified to {:.3f}'.format(
        args.new_entropy_loss))
    config.memory_net_config.new_cluster_entropy_loss = args.new_entropy_loss

  if args.new_cluster_loss >= 0.0:
    log.info('New cluster loss modified to {:.3f}'.format(
        args.new_cluster_loss))
    config.memory_net_config.new_cluster_loss = args.new_cluster_loss

  if args.proj_nlayer >= 0:
    log.info('Project nlayer modified to {}'.format(args.proj_nlayer))
    config.memory_net_config.cluster_projection_nlayer = args.proj_nlayer

  if args.siam_loss >= 0.0:
    log.info('Siam loss modified to {}'.format(args.siam_loss))
    config.memory_net_config.siam_loss = args.siam_loss

  if args.decay >= 0.0:
    log.info('Decay modified to {:.3f}'.format(args.decay))
    config.memory_net_config.decay = args.decay

  if args.schedule >= 0:
    log.info('Linear schedule modified to {:.3f}'.format(args.schedule))
    config.memory_net_config.linear_schedule = args.schedule

  # Modify optimization config.
  if config.optimizer_config.lr_scaling == "linear":
    for i in range(len(config.optimizer_config.lr_decay_steps)):
      print(len(gpus))
      print(gpus)
      config.optimizer_config.lr_decay_steps[i] //= len(gpus)
    config.optimizer_config.max_train_steps //= len(gpus)

    # Linearly scale learning rate.
    for i in range(len(config.optimizer_config.lr_list)):
      config.optimizer_config.lr_list[i] *= float(len(gpus))

  if 'SLURM_JOB_ID' in os.environ:
    log.info('SLURM job ID: {}'.format(os.environ['SLURM_JOB_ID']))

  if not args.reeval:
    # Build model.
    # model = build_pretrain_net(config)
    # mem_model = build_net(config, backbone=model.backbone)
    mem_model = build_net(config, distributed=hvd.size() > 1)
    reload_flag = None
    restore_steps = 0

    if is_chief:
      # Create save folder.
      save_folder = os.path.join(env_config.results, env_config.dataset,
                                 args.tag)
      if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

      # Checkpoint folder.
      ckpt_path = env_config.checkpoint
      if len(ckpt_path) > 0 and os.path.exists(ckpt_path):
        ckpt_folder = os.path.join(ckpt_path, os.environ['SLURM_JOB_ID'])
      else:
        ckpt_folder = save_folder

      # Reload previous checkpoint.
      log.info('Checkpoint folder {}'.format(ckpt_folder))
      if os.path.exists(ckpt_folder) and not args.eval:
        latest = latest_file(ckpt_folder, 'weights-')
        if latest is not None:
          log.info('Checkpoint already exists. Loading from {}'.format(latest))
          mem_model.load(latest)  # Not loading optimizer weights here.
          # assert mem_model.step.numpy() > 1, str(mem_model.step.numpy())
          reload_flag = latest
          restore_steps = int(reload_flag.split('-')[-1])
        else:
          latest = latest_file(save_folder, 'weights-')
          if latest is not None:
            log.info(
                'Checkpoint already exists. Loading from {}'.format(latest))
            mem_model.load(latest)  # Not loading optimizer weights here.
            # assert mem_model.step.numpy() > 1, str(mem_model.step.numpy())
            reload_flag = latest
            restore_steps = int(reload_flag.split('-')[-1])
      if not args.eval:
        save_config(config, save_folder)

      # Create TB logger.
      writer = tf.summary.create_file_writer(save_folder)
      logger = ExperimentLogger(writer)
    else:
      writer = None
      logger = None
      ckpt_folder = None
      save_folder = None

    # Get dataset.
    dataset = get_data_fs(env_config, load_train=True, load_test=args.eval)

    # Get data iterators.
    if env_config.dataset in ["matterport", "roaming-rooms"]:
      data = get_dataiter_sim(
          dataset,
          data_config,
          batch_size=config.optimizer_config.batch_size,
          nchw=mem_model.backbone.config.data_format == 'NCHW',
          seed=args.seed + restore_steps,
          distributed=len(gpus) > 1,
          same_seed=args.same_seed,
          iid=args.iid)
    elif env_config.dataset in ["say-cam"]:
      data = get_dataiter_vid(
          dataset,
          data_config,
          batch_size=config.optimizer_config.batch_size,
          nchw=mem_model.backbone.config.data_format == 'NCHW',
          seed=args.seed + restore_steps)
      # Not used.
      data['trainval_fs'] = None
      data['val_fs'] = None
      data['test_fs'] = None
    else:
      data = get_dataiter_continual(
          dataset,
          data_config,
          batch_size=config.optimizer_config.batch_size,
          nchw=mem_model.backbone.config.data_format == 'NCHW',
          save_additional_info=True,
          random_box=data_config.random_box,
          seed=args.seed + restore_steps)

  # Load model, training loop.
  if not args.eval:
    if args.pretrain is not None and reload_flag is None:
      mem_model.load(latest_file(args.pretrain, 'weights-'))
      if config.freeze_backbone:
        mem_model.backbone.set_trainable(False)  # Freeze the network.
        log.info('Backbone network is now frozen')
    with writer.as_default() if writer is not None else dcm() as gs:
      train(
          mem_model,
          data['train_fs'],
          data['trainval_fs'],
          data['val_fs'],
          ckpt_folder,
          final_save_folder=save_folder,
          maxlen=data_config.maxlen,
          logger=logger,
          writer=writer,
          is_chief=is_chief,
          reload_flag=reload_flag,
          unsup=data_config.unsupervised)
  else:
    results_file = os.path.join(save_folder, 'results.pkl')
    logfile = os.path.join(save_folder, 'results.csv')
    if os.path.exists(results_file) and args.reeval:
      # Re-display results.
      results_all = pkl.load(open(results_file, 'rb'))
      for split, name in zip(['trainval_fs', 'val_fs', 'test_fs'],
                             ['Train', 'Val', 'Test']):
        stats = get_stats(results_all[split], tmax=data_config.maxlen)
        log_results(stats, prefix=name, filename=logfile)
    else:
      # Load the most recent checkpoint.
      if args.usebest:
        latest = latest_file(save_folder, 'best-')
      else:
        latest = latest_file(save_folder, 'weights-')

      if latest is not None:
        mem_model.load(latest)
      else:
        if args.pretrain is not None:
          latest = latest_file(args.pretrain, 'weights-')
          if latest is not None:
            mem_model.load(latest)
          else:
            raise ValueError('Checkpoint not found')
        # else:
        #   raise ValueError('Checkpoint not found')
      data['trainval_fs'].reset()
      data['val_fs'].reset()
      data['test_fs'].reset()

      results_all = {}
      if args.testonly:
        split_list = ['test_fs']
        name_list = ['Test']
        nepisode_list = [config.num_episodes]
        # nepisode_list = [10]
      elif args.valonly:
        split_list = ['val_fs']
        name_list = ['Val']
        nepisode_list = [config.num_episodes]
      else:
        split_list = ['trainval_fs', 'val_fs', 'test_fs']
        name_list = ['Train', 'Val', 'Test']
        nepisode_list = [600, config.num_episodes, config.num_episodes]
        # nepisode_list = [10, 10, 10]

      for split, name, N in zip(split_list, name_list, nepisode_list):
        data[split].reset()
        # Need to reload checkpoints in between to prevent contamination.
        # N = 1
        if args.select_threshold:
          if args.thresh_compat:
            dist_q = select_threshold_compat(mem_model, data[split], 10)
            thresh = 0.5
          else:
            thresh = select_threshold(mem_model, data[split], 10)
            dist_q = None
        else:
          dist_q = None
          thresh = 0.5
        data[split].reset()
        r1 = evaluate(
            mem_model,
            data[split],
            N,
            verbose=True,
            reload=True,
            dist_q=dist_q,
            threshold=thresh)
        if data_config.unsupervised:
          stats = get_stats_unsup(r1)
          log_results_unsup(stats, prefix=name, filename=logfile)
        else:
          stats = get_stats(r1, tmax=data_config.maxlen)
          log_results(stats, prefix=name, filename=logfile)
        results_all[split] = r1


if __name__ == '__main__':
  print('hey1')
  parser = argparse.ArgumentParser(description='Lifelong Few-Shot Training')
  parser.add_argument('--config', type=str, default=None)
  parser.add_argument('--data', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--reeval', action='store_true')
  parser.add_argument('--pretrain', type=str, default=None)
  parser.add_argument('--tag', type=str, default=None)
  parser.add_argument('--testonly', action='store_true')
  parser.add_argument('--valonly', action='store_true')
  parser.add_argument('--usebest', action='store_true')
  parser.add_argument('--seed', type=int, default=0)
  # temporary flags for hparam tuning
  parser.add_argument('--new_prob_target', type=float, default=-1.0)
  parser.add_argument('--new_cluster_thresh', type=float, default=-1.0)
  parser.add_argument('--max_classes', type=int, default=-1)
  parser.add_argument('--area_lb', type=float, default=-1.0)
  parser.add_argument('--dist_q', type=float, default=-1.0)
  parser.add_argument('--num_aug', type=int, default=-1)
  parser.add_argument('--entropy_loss', type=float, default=-1.0)
  parser.add_argument('--soft_label_temp', type=float, default=-1.0)
  parser.add_argument('--new_entropy_loss', type=float, default=-1.0)
  parser.add_argument('--new_cluster_loss', type=float, default=-1.0)
  parser.add_argument('--proj_nlayer', type=int, default=-1)
  parser.add_argument('--siam_loss', type=float, default=-1.0)
  parser.add_argument('--random_rotate', type=float, default=-1.0)
  parser.add_argument('--maxlen', type=int, default=-1)
  parser.add_argument('--decay', type=float, default=-1.0)
  parser.add_argument('--schedule', type=int, default=-1)
  parser.add_argument('--select_threshold', action='store_true')
  parser.add_argument('--large_sweep', action='store_true')
  parser.add_argument('--imbalance_ratio', type=float, default=-1.0)
  parser.add_argument('--steps_per_val', type=int, default=-1)
  parser.add_argument('--frame_rate', type=int, default=-1)
  parser.add_argument('--thresh_compat', action='store_true')
  parser.add_argument('--same_seed', action='store_true')
  parser.add_argument('--iid', action='store_true')
  args = parser.parse_args()
  tf.random.set_seed(1234)
  main()
