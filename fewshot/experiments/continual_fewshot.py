"""
Train an continual few-shot network.
Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os

import numpy as np
import six
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
import horovod.tensorflow as hvd

from google.protobuf.text_format import Merge, MessageToString

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(
      gpus[hvd.local_rank() % len(gpus)], 'GPU')
is_chief = hvd.rank() == 0
from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_net, build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter
from fewshot.experiments.get_data_iter import get_dataiter_continual
from fewshot.experiments.get_stats import get_stats, get_stats_unsup
from fewshot.experiments.oc_fewshot import evaluate
from fewshot.experiments.utils import get_data
from fewshot.experiments.utils import (ExperimentLogger, delete_old_ckpt,
                                       get_config, get_data_fs, latest_file,
                                       save_config)
from fewshot.utils.dummy_context_mgr import dummy_context_mgr as dcm
from fewshot.utils.logger import get as get_logger
from tqdm import tqdm

log = get_logger()


def train(model,
          dataiter_list,
          dataitertest_list_list,
          model_eval_list,
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
  N = model.config.optimizer_config.max_train_steps
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

  start_stage = start // N
  start_step = start % N
  Ntotal = N * len(dataiter_list)

  for s, dataiter in enumerate(dataiter_list):
    if s < start_stage:
      continue
    elif s == start_stage:
      it = six.moves.xrange(start_step, N)
    else:
      it = six.moves.xrange(N)

    if is_chief:
      it = tqdm(it, ncols=0, desc="stage {} training".format(s))

    for i, batch in zip(it, dataiter):
      tf.summary.experimental.set_step(s * N + i + 1)

      if args.mini_batch:
        x = batch['x']
        y = batch['y']
        kwargs = {}
      else:
        x = batch['x_s']
        y = batch['y_s']
        y_gt = batch['y_gt']
        if args.dist_q > 0.0:
          dist_q = tf.constant(args.dist_q)
        else:
          dist_q = None
        kwargs = {'y_gt': y_gt, 'flag': batch['flag_s'], 'dist_q': dist_q}
        kwargs['writer'] = writer
      loss = model.train_step(x, y, **kwargs)

      if i == start_step and s == start_stage and reload_flag is not None:
        print(reload_flag)
        model.load(reload_flag, load_optimizer=True)

      # # Synchronize distributed weights.
      # if i == start_step and s == start_stage and model._distributed:
      #   hvd.broadcast_variables(model.var_to_optimize(), root_rank=0)
      #   hvd.broadcast_variables(model.optimizer.variables(), root_rank=0)
      #   if model.config.set_backbone_lr:
      #     hvd.broadcast_variables(model._bb_optimizer.variables(),
      # root_rank=0)

      step = model.step.numpy()

      if is_chief and (step % config.steps_per_val == 0 or step == 1):

        eval_results = []
        for dataitertest_list, model_eval in zip(dataitertest_list_list,
                                                 model_eval_list):
          eval_results_ = []
          eval_list = tqdm(dataitertest_list, ncols=0, desc="eval")
          for j, dataiter_eval in enumerate(eval_list):
            dataiter_eval.reset()
            if dataiter_eval.config.unsupervised:
              dist_q = select_threshold(model_eval, dataiter_eval, 10)
              r1 = evaluate(model_eval, dataiter_eval, 60, dist_q=dist_q)
              eval_results_.append(get_stats_unsup(r1))
              try_log('continual/mi stage{}'.format(j), step,
                      eval_results_[-1]['mutual_info'] * 100.0)
            else:
              r1 = evaluate(model_eval, dataiter_eval, 60)
              eval_results_.append(
                  get_stats(r1, nshot_max=nshot_max, tmax=maxlen))
              try_log('continual/ap stage{}'.format(j), step,
                      eval_results_[-1]['ap'] * 100.0)
          eval_results.append(eval_results_)
        try_log('lr', step, model.learn_rate())
        print()

      # Save.
      if is_chief and ((i + 1) % config.steps_per_save == 0):
        model.save(os.path.join(ckpt_folder, 'weights-{}'.format(step)))

        # Delete earlier checkpoint.
        if not save_all and ckpt_folder == final_save_folder:
          delete_old_ckpt(ckpt_folder, "weights-", step, config.steps_per_save)

      # Write logs.
      if is_chief and ((i + 1) % config.steps_per_log == 0):
        try_log('loss/all', step, loss)
        try_flush()

        # Update progress bar.
        post_fix_dict = {}
        post_fix_dict['lr'] = '{:.3e}'.format(model.learn_rate())
        post_fix_dict['loss'] = '{:.3e}'.format(loss)
        if r is not None:
          if unsup:
            post_fix_dict['mi'] = '{:.3f}'.format(
                eval_results[0][s]['mutual_info'] * 100.0)
          else:
            post_fix_dict['ap'] = '{:.3f}'.format(
                eval_results[1][s]['ap'] * 100.0)
        it.set_postfix(**post_fix_dict)

  # Save.
  if is_chief and final_save_folder is not None:
    model.save(os.path.join(final_save_folder, 'weights-{}'.format(Ntotal)))


def select_threshold(model, data, nepisode, unsup=True):
  dist_q_list = np.linspace(0.0, 0.1, 25).astype(np.float32)
  # N = 1
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
      # print('dist', dist_q, 'MI', stats['mutual_info'])
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
  config.num_steps = data_config.maxlen

  if args.max_classes > 0:
    # Manual case.
    if not args.mini_batch:
      config.num_classes = args.max_classes  # Assign num classes.
      data_config.unk_id = args.max_classes
    config.memory_net_config.max_classes = args.max_classes
  elif config.memory_net_config.create_unk:
    # Unsupervised case.
    if not args.mini_batch:
      config.num_classes = data_config.maxlen  # Assign num classes.
      data_config.unk_id = data_config.maxlen
    config.memory_net_config.max_classes = data_config.maxlen
  else:
    # Standard supervised case.
    if not args.mini_batch:
      config.num_classes = data_config.nway  # Assign num classes.
      data_config.unk_id = data_config.nway
    config.memory_net_config.max_classes = data_config.nway

  # config.memory_net_config.max_stages = data_config.nstage
  config.memory_net_config.max_items = data_config.maxlen
  config.oml_config.num_classes = data_config.nway
  config.fix_unknown = data_config.fix_unknown  # Assign fix unknown ID.
  log.info('Number of classes {}'.format(data_config.nway))
  log.info('Number of memory classes {}'.format(
      config.memory_net_config.max_classes))

  # By pass config.
  if args.newprob_target > 0.0:
    log.info('New cluster thresh modified to {:.3f}'.format(
        args.newprob_target))
    config.memory_net_config.new_cluster_thresh = args.newprob_target

  if args.area_lb > 0.0:
    log.info('Area LB modified to {:.3f}'.format(args.area_lb))
    data_config.area_lb = args.area_lb

  if args.num_aug > 0:
    log.info('Num aug modified to {}'.format(args.num_aug))
    data_config.num_aug = args.num_aug

  if args.entropy_loss >= 0.0:
    log.info('Entropy loss modified to {:.3f}'.format(args.entropy_loss))
    config.memory_net_config.entropy_loss = args.entropy_loss

  if args.proj_nlayer >= 0:
    log.info('Project nlayer modified to {}'.format(args.proj_nlayer))
    config.memory_net_config.cluster_projection_nlayer = args.proj_nlayer

  if args.decay >= 0.0:
    log.info('Decay modified to {:.3f}'.format(args.decay))
    config.memory_net_config.decay = args.decay

  if args.schedule >= 0:
    log.info('Linear schedule modified to {:.3f}'.format(args.schedule))
    config.memory_net_config.linear_schedule = args.schedule

  # Modify optimization config.
  if config.optimizer_config.lr_scaling == "linear":
    for i in range(len(config.optimizer_config.lr_decay_steps)):
      config.optimizer_config.lr_decay_steps[i] //= len(gpus)
    config.optimizer_config.max_train_steps //= len(gpus)

    # Linearly scale learning rate.
    for i in range(len(config.optimizer_config.lr_list)):
      config.optimizer_config.lr_list[i] *= float(len(gpus))

  if 'SLURM_JOB_ID' in os.environ:
    log.info('SLURM job ID: {}'.format(os.environ['SLURM_JOB_ID']))

  if not args.reeval:
    if config.model_class == "pretrain_net":
      main_model = build_pretrain_net(config)
    else:
      main_model = build_net(config)

    # Build an unsup model and a sup model for evaluation.
    config_unsup = ExperimentConfig()
    Merge(MessageToString(config), config_unsup)
    config_unsup.model_class = "online_proto_net"
    config_unsup.memory_class = "proto_memory_v2"
    config_unsup.memory_net_config.create_unk = True
    config_unsup.num_classes = data_config.maxlen  # Assign num classes.
    config_unsup.memory_net_config.max_classes = data_config.maxlen
    model_unsup = build_net(config_unsup, backbone=main_model.backbone)

    config_sup = ExperimentConfig()
    Merge(MessageToString(config), config_sup)
    config_sup.model_class = "online_proto_net"
    config_sup.memory_class = "proto_memory_v2"
    config_sup.memory_net_config.create_unk = False
    config_sup.num_classes = data_config.nway  # Assign num classes.
    config_sup.memory_net_config.max_classes = data_config.nway
    model_sup = build_net(config_sup, backbone=main_model.backbone)

    # Build both supervised and unsupervised eval data.
    data_config_unsup = EpisodeConfig()
    Merge(MessageToString(data_config), data_config_unsup)
    data_config_unsup.unsupervised = True
    data_config_unsup.continual = True
    data_config_unsup.augmentation = False
    data_config_unsup.semisupervised = False
    data_config_unsup.unk_id = data_config.maxlen

    data_config_sup = EpisodeConfig()
    Merge(MessageToString(data_config), data_config_sup)
    data_config_sup.unsupervised = False
    data_config_sup.continual = True
    data_config_sup.augmentation = False
    data_config_sup.semisupervised = False
    data_config_sup.unk_id = data_config.nway

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
          main_model.load(latest)  # Not loading optimizer weights here.
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
    dataset = get_data(env_config)
    dataset_fs = get_data_fs(env_config, load_train=True, load_test=args.eval)

    # Get data iterators.
    if env_config.dataset in ["matterport", "roaming-rooms"]:
      assert False, "Not supported"
    else:
      data_list = []
      dataeval_list_unsup = []
      dataeval_list_sup = []

      # For training.
      if args.mini_batch:
        for stage in range(args.nstage):
          if args.iid:
            stage_ = None
            data_config.continual = False
          else:
            stage_ = stage
          data = get_dataiter(
              dataset,
              batch_size=config.optimizer_config.batch_size,
              nchw=main_model.backbone.config.data_format == 'NCHW',
              continual=not (stage_ is None),
              stage=stage_)
          data_list.append(data['train'])

      else:
        for stage in range(args.nstage):
          if args.iid:
            stage_ = None
            data_config.continual = False
          else:
            stage_ = stage
          data = get_dataiter_continual(
              dataset_fs,
              data_config,
              batch_size=1,
              nchw=main_model.backbone.config.data_format == 'NCHW',
              save_additional_info=True,
              random_box=data_config.random_box,
              seed=args.seed + restore_steps,
              stage=stage_)
          data_list.append(data['train_fs'])

      # For unsup eval.
      for stage in range(args.nstage):
        data = get_dataiter_continual(
            dataset_fs,
            data_config_unsup,
            batch_size=1,
            nchw=main_model.backbone.config.data_format == 'NCHW',
            save_additional_info=True,
            random_box=data_config.random_box,
            seed=args.seed,
            stage=stage)
        dataeval_list_unsup.append(data['trainval_fs'])

      # For sup eval.
      for stage in range(args.nstage):
        data = get_dataiter_continual(
            dataset_fs,
            data_config_sup,
            batch_size=1,
            nchw=main_model.backbone.config.data_format == 'NCHW',
            save_additional_info=True,
            random_box=data_config.random_box,
            seed=args.seed,
            stage=stage)
        dataeval_list_sup.append(data['trainval_fs'])

      dataeval_list_list = [dataeval_list_unsup, dataeval_list_sup]
      model_eval_list = [model_unsup, model_sup]

  # Load model, training loop.
  if not args.eval:
    if args.pretrain is not None and reload_flag is None:
      main_model.load(latest_file(args.pretrain, 'weights-'))
      if config.freeze_backbone:
        main_model.backbone.set_trainable(False)  # Freeze the network.
        log.info('Backbone network is now frozen')
    with writer.as_default() if writer is not None else dcm() as gs:
      train(
          main_model,
          data_list,
          dataeval_list_list,
          model_eval_list,
          ckpt_folder,
          final_save_folder=save_folder,
          maxlen=data_config.maxlen,
          logger=logger,
          writer=writer,
          is_chief=is_chief,
          reload_flag=reload_flag,
          unsup=data_config.unsupervised,
          save_all=args.save_all)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Continual Few-Shot Training')
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
  parser.add_argument('--newprob_target', type=float, default=-1.0)
  parser.add_argument('--max_classes', type=int, default=-1)
  parser.add_argument('--area_lb', type=float, default=-1.0)
  parser.add_argument('--dist_q', type=float, default=-1.0)
  parser.add_argument('--num_aug', type=int, default=-1)
  parser.add_argument('--entropy_loss', type=float, default=-1.0)
  parser.add_argument('--proj_nlayer', type=int, default=-1)
  parser.add_argument('--decay', type=float, default=-1.0)
  parser.add_argument('--schedule', type=int, default=-1)
  parser.add_argument('--select_threshold', action='store_true')
  parser.add_argument('--nstage', type=int, default=10)
  parser.add_argument('--iid', action='store_true')
  parser.add_argument('--save_all', action='store_true')
  # Mini-batch iterator instead of episodic CRP.
  parser.add_argument('--mini_batch', action='store_true')
  args = parser.parse_args()
  tf.random.set_seed(1234)
  main()
