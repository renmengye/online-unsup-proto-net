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
import scipy.stats

tf.get_logger().setLevel('ERROR')
import horovod.tensorflow as hvd

# from google.protobuf.text_format import Merge, MessageToString

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(
      gpus[hvd.local_rank() % len(gpus)], 'GPU')
is_chief = hvd.rank() == 0
from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_net
from fewshot.experiments.get_data_iter import get_dataiter, get_dataiter_contrastive  # NOQA
from fewshot.experiments.utils import get_data
from fewshot.experiments.utils import (ExperimentLogger, delete_old_ckpt,
                                       get_config, get_data_fs, latest_file,
                                       save_config)
from fewshot.utils.dummy_context_mgr import dummy_context_mgr as dcm
from fewshot.utils.logger import get as get_logger
from fewshot.experiments.metrics import stderr
from tqdm import tqdm

log = get_logger()


def evaluate_knn(model, dataiter):
  dataiter.reset()
  it = six.moves.xrange(len(dataiter))
  results = []

  # Build nearest neighbor.
  h_list = []
  y_list = []

  for i, batch in zip(it, dataiter):
    x = batch['x']
    y = batch['y']

    if model.config.model_class.startswith('online'):
      x = x[None]

    # Run model.
    h = model.run_backbone(x, is_training=False)
    h_list.append(h)
    y_list.append(y)

  k_list = [3, 5, 10]
  with tf.device('/cpu:0'):
    h_list = tf.concat(h_list, axis=1)[0]  # [N, D]
    y_list = tf.concat(y_list, axis=0)  # [N]
    # np.dot(h_list, h_list.T)  # [N, N]
    h_list = tf.math.l2_normalize(h_list, axis=-1)
    sim = tf.matmul(h_list, h_list, transpose_b=True)  # [N, N]
    N = h_list.shape[0]
    sim = sim - 2 * tf.eye(N)  # Suppress diagonals.

    sim_idx = tf.argsort(sim, axis=-1, direction='DESCENDING')  # [N, N]
    answer = tf.gather(y_list, sim_idx)  # [N, N]
  y_np = y_list.numpy()
  results = {}
  for k in [3, 5, 10]:
    a = scipy.stats.mode(answer[:, :k].numpy(), axis=1)  # Majority voting.
    correct = np.equal(a.mode[:, 0], y_np).astype(np.float32)
    acc = correct.mean()
    results['{}nnacc'.format(k)] = acc
    results['{}nnacc_se'.format(k)] = stderr(correct)

  return results


def train(model,
          dataiter,
          dataitertest,
          ckpt_folder,
          final_save_folder=None,
          logger=None,
          writer=None,
          is_chief=True,
          reload_flag=None):
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

  it = six.moves.xrange(start, N)
  if is_chief:
    it = tqdm(it, ncols=0)

  for i, batch in zip(it, dataiter):
    tf.summary.experimental.set_step(i + 1)
    x = batch['x']
    # y = batch['y']
    kwargs = {}
    kwargs['writer'] = writer
    if model.config.model_class.startswith('online'):
      x = x[None]  # Add a batch dimension.
      unk_id = model.config.memory_net_config.max_classes
      y = tf.zeros([1, x.shape[1]], dtype=tf.int32) + unk_id
      flag = tf.ones([1, x.shape[1]], dtype=tf.int64)
      if args.dist_q >= 0.0:
        dist_q = tf.constant(args.dist_q)
        # Warm up with 3000.
        # dist_q = tf.constant(
        #     np.minimum(3000, i) / 3000.0 * args.dist_q, dtype=tf.float32)
      else:
        dist_q = None
      loss = model.train_step(x, y, y_gt=y, flag=flag, dist_q=dist_q, **kwargs)
    else:
      loss = model.trian_step(x, **kwargs)

    if i == start and reload_flag is not None:
      print(reload_flag)
      model.load(reload_flag, load_optimizer=True)

    step = model.step.numpy()

    # Evaluate
    if is_chief and (step % config.steps_per_val == 0 or step == 1):
      try_log('lr', step, model.learn_rate())
      results = evaluate_knn(model, dataitertest)
      for k in results:
        try_log('nearest_neighbor/{}'.format(k), step, results[k] * 100.0)
      print()

    # Save.
    if is_chief and ((i + 1) % config.steps_per_save == 0):
      model.save(os.path.join(ckpt_folder, 'weights-{}'.format(step)))

      # Delete earlier checkpoint.
      if ckpt_folder == final_save_folder:
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
        post_fix_dict['1nn_acc'] = '{:.3f}'.format(results['1nn_acc'] * 100.0)
      it.set_postfix(**post_fix_dict)

  # Save.
  if is_chief and final_save_folder is not None:
    model.save(os.path.join(final_save_folder, 'weights-{}'.format(N)))


def main():
  assert tf.executing_eagerly(), 'Only eager mode is supported.'
  assert args.config is not None, 'You need to pass in model config file path'
  assert args.env is not None, 'You need to pass in environ config file path'
  assert args.tag is not None, 'You need to specify a tag'

  log.info('Command line args {}'.format(args))
  config = get_config(args.config, ExperimentConfig)
  # data_config = get_config(args.data, EpisodeConfig)
  env_config = get_config(args.env, EnvironmentConfig)
  log.info('Model: \n{}'.format(config))
  # log.info('Data episode: \n{}'.format(data_config))
  log.info('Environment: \n{}'.format(env_config))
  # config.num_classes = data_config.maxlen  # TODO: change this.
  # config.num_steps = data_config.maxlen
  config.num_steps = config.optimizer_config.batch_size

  if args.max_classes > 0:
    config.memory_net_config.max_classes = args.max_classes

  # config.memory_net_config.max_stages = data_config.nstage
  config.memory_net_config.max_items = config.optimizer_config.batch_size
  config.fix_unknown = True  # Assign fix unknown ID.
  log.info('Number of memory classes {}'.format(
      config.memory_net_config.max_classes))

  # By pass config.
  if args.newprob_target > 0.0:
    log.info('New cluster thresh modified to {:.3f}'.format(
        args.newprob_target))
    config.memory_net_config.new_cluster_thresh = args.newprob_target

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
    model = build_net(config)

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
          model.load(latest)  # Not loading optimizer weights here.
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

    if args.contrastive:
      train_data = get_dataiter_contrastive(
          dataset,
          batch_size=args.batch_size,
          nchw=model.backbone.config.data_format == 'NCHW',
          distributed=False,
          color_distort=env_config.random_color,
          flip=env_config.random_flip,
          color_distort_strength=env_config.random_color_strength,
          area_range_lb=env_config.area_range_lb,
          min_object_covered=0.1,
          num_views=2,
          aug=model.config.model_class.startswith('online'),
          cycle=True,
          seed=0)['train']
      test_data = get_dataiter(
          dataset,
          batch_size=args.batch_size,
          nchw=model.backbone.config.data_format == 'NCHW',
          continual=False)['test']
    else:
      data = get_dataiter(
          dataset,
          batch_size=args.batch_size,
          nchw=model.backbone.config.data_format == 'NCHW',
          continual=False)
      train_data = data['train']
      test_data = data['test']

  # Load model, training loop.
  if not args.eval:
    if args.pretrain is not None and reload_flag is None:
      model.load(latest_file(args.pretrain, 'weights-'))
      if config.freeze_backbone:
        model.backbone.set_trainable(False)  # Freeze the network.
        log.info('Backbone network is now frozen')
    with writer.as_default() if writer is not None else dcm() as gs:
      train(
          model,
          train_data,
          test_data,
          ckpt_folder,
          final_save_folder=save_folder,
          logger=logger,
          writer=writer,
          is_chief=is_chief,
          reload_flag=reload_flag)

  # Load the most recent checkpoint.
  if args.usebest:
    latest = latest_file(save_folder, 'best-')
  else:
    latest = latest_file(save_folder, 'weights-')

  if latest is not None:
    model.load(latest)
  else:
    if args.pretrain is not None:
      latest = latest_file(args.pretrain, 'weights-')
      if latest is not None:
        model.load(latest)
      else:
        raise ValueError('Checkpoint not found')
  results = evaluate_knn(model, test_data)
  print(results)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Continual Few-Shot Training')
  parser.add_argument('--config', type=str, default=None)
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
  parser.add_argument('--max_classes', type=int, default=150)
  parser.add_argument('--area_lb', type=float, default=-1.0)
  parser.add_argument('--dist_q', type=float, default=-1.0)
  parser.add_argument('--num_aug', type=int, default=-1)
  parser.add_argument('--entropy_loss', type=float, default=-1.0)
  parser.add_argument('--proj_nlayer', type=int, default=-1)
  parser.add_argument('--decay', type=float, default=-1.0)
  parser.add_argument('--schedule', type=int, default=-1)
  parser.add_argument('--select_threshold', action='store_true')
  # Mini-batch iterator instead of episodic CRP.
  parser.add_argument('--contrastive', action='store_true')
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--fs', action='store_true')
  args = parser.parse_args()
  tf.random.set_seed(1234)
  main()
