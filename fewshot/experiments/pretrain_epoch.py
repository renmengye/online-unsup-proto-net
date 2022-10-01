"""
Pretrain a network on regular classification.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
  pretrain.py --config [CONFIG] --tag [TAG} --dataset [DATASET] \
              --data_folder [DATA FOLDER] --results [SAVE FOLDER]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import glob
import os

import six

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf

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
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter_epoch, get_dataiter_matterport_category
from fewshot.experiments.utils import (ExperimentLogger, get_config, get_data,
                                       latest_file, save_config)
from fewshot.utils.dummy_context_mgr import dummy_context_mgr as dcm
from fewshot.utils.logger import get as get_logger
from tqdm import tqdm

log = get_logger()


def label_equal(pred, label, axis=-1):
  return pred == label.astype(pred.dtype)


def top1_correct(pred, label, axis=-1):
  """Calculates top 1 correctness."""
  assert pred.shape[0] == label.shape[0], '{} != {}'.format(
      pred.shape[0], label.shape[0])
  pred_idx = np.argmax(pred, axis=axis)
  return pred_idx == label.astype(pred_idx.dtype)


def top1_acc(pred, label, axis=-1):
  """Calculates top 1 accuracy."""
  return top1_correct(pred, label, axis=axis).mean()


def topk_acc(pred, label, k, axis=-1):
  """Calculates top 5 accuracy."""
  assert pred.shape[0] == label.shape[0], '{} != {}'.format(
      pred.shape[0], label.shape[0])
  topk_choices = np.argsort(pred, axis=axis)
  if len(topk_choices.shape) == 2:
    topk_choices = topk_choices[:, ::-1][:, :k]
  elif len(topk_choices.shape) == 3:
    topk_choices = topk_choices[:, :, ::-1][:, :, :k]
  else:
    raise NotImplementedError()
  return np.sum(topk_choices == np.expand_dims(label, axis), axis=axis).mean()


def stderr(array, axis=0):
  """Calculates standard error."""
  return array.std(axis=axis) / np.sqrt(float(array.shape[0]))


def evaluate(model, dataiter, num_steps, verbose=False):
  """Evaluates accuracy."""
  acc_list = []
  acc_top5_list = []
  batch_size = model.config.optimizer_config.batch_size
  if num_steps <= 0:
    # Determine the total number of steps manually.
    num_steps = len(dataiter)
  if verbose:
    it = tqdm(range(num_steps), ncols=0)
  else:
    it = range(num_steps)
  for ii, batch in zip(it, dataiter):
    x = batch['x']
    y = batch['y']
    prediction_a = model.eval_step(x).numpy()
    acc_list.append(top1_acc(prediction_a, y.numpy()))
    acc_top5_list.append(topk_acc(prediction_a, y.numpy(), 5))
    if verbose:
      it.set_postfix(acc=u'{:.3f}±{:.3f}'.format(
          np.array(acc_list).mean() * 100.0,
          stderr(np.array(acc_list)) * 100.0))
  acc_list = np.array(acc_list)
  acc_top5_list = np.array(acc_top5_list)
  results_dict = {
      'acc': acc_list.mean(),
      'acc_se': stderr(acc_list),
      'acc_top5': acc_top5_list.mean(),
      'acc_top5_se': stderr(acc_top5_list)
  }
  return results_dict


def train(model,
          dataiter,
          dataiter_test,
          ckpt_folder,
          final_save_folder,
          logger,
          writer,
          reload_flag=None,
          save_all=False):
  """Trains the model."""
  Nsteps = len(dataiter)
  Nepoch = model.config.optimizer_config.max_train_epochs
  config = model.config.train_config
  start = model.step.numpy()
  if start > 0:
    log.info('Restore from step {}'.format(start))
  estart = start // Nsteps
  sstart = start % Nsteps
  rtrain = None

  with writer.as_default() if writer is not None else dcm() as gs:
    for e in range(estart, Nepoch):
      dataiter.reset()
      if e == estart:
        it = six.moves.xrange(sstart, Nsteps)
      else:
        it = six.moves.xrange(Nsteps)
      it = tqdm(it, ncols=0)
      for ii, batch in zip(it, dataiter):
        x = batch['x']
        y = batch['y']
        loss = model.train_step(x, y)
        step = model.step.numpy()

        # Reload model weights.
        if ii == start and reload_flag is not None:
          model.load(reload_flag, load_optimizer=True)

        # Horovod init.
        if ii == sstart and e == estart and len(gpus) > 1:
          hvd.broadcast_variables(model.var_to_optimize(), root_rank=0)
          hvd.broadcast_variables(model.optimizer.variables(), root_rank=0)
          if model.config.set_backbone_lr:
            hvd.broadcast_variables(
                model._bb_optimizer.variables(), root_rank=0)

        # Save.
        if step % config.steps_per_save == 0 and is_chief:
          model.save(os.path.join(ckpt_folder, 'weights-{}'.format(step)))

          # Delete earlier checkpoint.
          if not save_all and ckpt_folder == final_save_folder:
            fnames = glob.glob(os.path.join(ckpt_folder, 'weights-*'))
            for f in fnames:
              name = f.split('-')[-1].split('.')[0]
              if step - int(name) > 5 * config.steps_per_save:
                os.remove(f)
          else:
            fnames = glob.glob(os.path.join(final_save_folder, 'weights-*'))
            for f in fnames:
              name = f.split('-')[-1].split('.')[0]
              if step - int(name) >= config.steps_per_save:
                os.remove(f)

        # Evaluate.
        if step % config.steps_per_val == 0 and is_chief:
          rtrain = evaluate(model, dataiter, 100)
          logger.log('acc train', step, rtrain['acc'] * 100.0)
          logger.log('lr', step, model.learn_rate())

          if dataiter_test is not None:
            dataiter_test.reset()
            rtest = evaluate(model, dataiter_test, 100)
            logger.log('acc val', step, rtest['acc'] * 100.0)
          print()

        # Write logs.
        if step % config.steps_per_log == 0 and is_chief:
          logger.log('loss', step, loss)
          logger.flush()

          # Update progress bar.
          post_fix_dict = {}
          if rtrain is not None:
            post_fix_dict['acc_t'] = '{:.1f}'.format(rtrain['acc'] * 100.0)
            if dataiter_test is not None:
              post_fix_dict['acc_v'] = '{:.1f}'.format(rtest['acc'] * 100.0)
          post_fix_dict['lr'] = '{:.1e}'.format(model.learn_rate())
          post_fix_dict['loss'] = '{:.1e}'.format(loss)
          it.set_postfix(**post_fix_dict)

  # Save.
  if is_chief and final_save_folder is not None:
    model.save(os.path.join(final_save_folder, 'weights-{}.pkl'.format(step)))


def log_results(results, prefix=None, filename=None):
  """Log results to a file."""
  acc = results['acc'] * 100.0
  se = results['acc_se'] * 100.0
  name = prefix if prefix is not None else 'Acc'
  if filename is not None:
    with open(filename, 'a') as f:
      f.write('{}\t\t{:.3f}\t\t{:.3f}\n'.format(name, acc, se))
  log.info(u'{} Acc = {:.3f} ± {:.3f}'.format(name, acc, se))


def main():
  assert tf.executing_eagerly()
  assert args.config is not None, 'You need to pass in model config file path'
  assert args.env is not None, 'You need to pass in the env config file path'
  assert args.tag is not None, 'You need to specify a tag'
  print(args)
  config = get_config(args.config, ExperimentConfig)
  if args.data is not None:
    data_config = get_config(args.data, EpisodeConfig)
  env_config = get_config(args.env, EnvironmentConfig)
  model = build_pretrain_net(config)
  dataset = get_data(env_config)
  log.info('Frame rate = 20')
  if args.online:
    frame_rate = 20
    shuffle = False
  else:
    frame_rate = 1
    shuffle = True

  if env_config.dataset in ['matterport', 'roaming-rooms']:
    for d in dataset.keys():
      if d != 'metadata':
        dataset[d]._train_size = 128000  # Smaller.

    data = get_dataiter_matterport_category(
        dataset,
        config.optimizer_config.batch_size,
        nchw=model.backbone.config.data_format == "NCHW")
  else:
    # Say-CAM time dataset.
    data = get_dataiter_epoch(
        dataset,
        config.optimizer_config.batch_size,
        nchw=model.backbone.config.data_format == 'NCHW',
        data_aug=True,
        use_file=True,
        frame_rate=frame_rate,
        shuffle=shuffle)

  if is_chief:
    save_folder = os.path.join(env_config.results, env_config.dataset,
                               args.tag)
    ckpt_path = env_config.checkpoint
    if len(ckpt_path) > 0 and os.path.exists(ckpt_path):
      ckpt_folder = os.path.join(ckpt_path, os.environ['SLURM_JOB_ID'])
      log.info('Checkpoint folder: {}'.format(ckpt_folder))
    else:
      ckpt_folder = save_folder

    latest = None
    prefix = 'weights-'
    if os.path.exists(ckpt_folder):
      latest = latest_file(ckpt_folder, prefix)

    if latest is None and os.path.exists(save_folder):
      latest = latest_file(save_folder, prefix)

    if latest is not None:
      log.info('Checkpoint already exists. Loading from {}'.format(latest))
      model.load(latest)  # Not loading optimizer weights here.
      reload_flag = latest
      restore_steps = int(reload_flag.split('-')[-1].split('.')[0])
      model.step.assign(restore_steps)

    save_config(config, save_folder)
    writer = tf.summary.create_file_writer(save_folder)
    logger = ExperimentLogger(writer)

    latest = None
    reload_flag = None
    if os.path.exists(ckpt_folder):
      latest = latest_file(ckpt_folder, 'weights-')

    if latest is None and os.path.exists(save_folder):
      latest = latest_file(save_folder, 'weights-')

    if not os.path.exists(save_folder):
      save_config(config, save_folder)

    if latest is not None:
      log.info('Checkpoint already exists. Loading from {}'.format(latest))
      model.load(latest)  # Not loading optimizer weights here.
      reload_flag = latest
      restore_steps = int(reload_flag.split('-')[-1].split('.')[0])
      model.step.assign(restore_steps)
  else:
    save_folder = None
    ckpt_folder = None
    writer = None
    logger = None
    reload_flag = None

  if not args.eval:
    train(
        model,
        data['train'],
        data['val'],
        ckpt_folder,
        save_folder,
        logger,
        writer,
        reload_flag=reload_flag,
        save_all=False)
  else:
    # Load the most recent checkpoint.
    if args.readout:
      prefix = 'readout-'
    else:
      prefix = 'weights-'
    model.load(latest_file(save_folder, prefix))

  logfile = os.path.join(save_folder, 'results.tsv')
  for split, name in zip(['test'], ['Test']):
    if split in data and data[split] is not None:
      data[split].reset()
      r = evaluate(model, data[split], -1)
      log_results(r, prefix=name, filename=logfile)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pretrain')
  parser.add_argument('--config', type=str, default=None)
  parser.add_argument('--data', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--tag', type=str, default=None)
  parser.add_argument('--online', action='store_true')
  parser.add_argument('--matterport', action='store_true')
  parser.add_argument('--readout', action='store_true')
  args = parser.parse_args()
  main()
