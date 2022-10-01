"""
Pretrain the network using an unsupervised representation algorithm.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
  pretrain_contrastive.py --config [CONFIG] --tag [TAG} --dataset [DATASET] \
              --data_folder [DATA FOLDER] --results [SAVE FOLDER]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import glob
import six
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(
      gpus[hvd.local_rank() % len(gpus)], 'GPU')
is_chief = hvd.rank() == 0

from tqdm import tqdm

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_contrastive_net
from fewshot.experiments.build_model import build_fewshot_net
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter_contrastive
from fewshot.experiments.get_data_iter import get_dataiter_fewshot
from fewshot.experiments.get_data_iter import get_dataiter_epoch
from fewshot.experiments.utils import ExperimentLogger, delete_old_ckpt
from fewshot.experiments.utils import save_config
from fewshot.experiments.utils import get_config
from fewshot.experiments.utils import get_data
from fewshot.experiments.utils import get_data_fs
from fewshot.experiments.utils import latest_file
from fewshot.experiments.pretrain import evaluate_fs
from fewshot.utils.logger import get as get_logger
from fewshot.utils.dummy_context_mgr import dummy_context_mgr
from fewshot.experiments.pretrain import top1_acc

log = get_logger()


def visualize_image(x, ncols, cv2=False):
  """Reshape x [N C H W] in to an array of images."""
  C, H, W = x.shape[1], x.shape[2], x.shape[3]
  x = tf.transpose(x, [0, 2, 3, 1])  # [N, H, W, C]
  # x = x / 2.0 + 0.5  # Unnormalize
  x = tf.reshape(x, [-1, ncols, H, W, C])
  P = 5
  x = tf.pad(
      x, [[0, 0], [0, 0], [P, P], [P, P], [0, 0]],
      "CONSTANT",
      constant_values=1.0)
  x = tf.transpose(x, [0, 2, 1, 3, 4])  # [Row, H, Col, W, C]
  x = tf.reshape(x, [1, -1, ncols * (W + 2 * P), C])
  if cv2:
    x = x[0].numpy()
    x = (x * 255).astype(np.uint8)
  return x


def get_evaluate_nn_fn(model,
                       data_train,
                       data_val,
                       data_test,
                       is_training=False,
                       rgb=False):
  """Evaluate using nearest neighbor."""

  def _eval_fn(step):
    data_train.reset()
    it = range(len(data_train))
    it = tqdm(it, ncols=0, desc='train')
    # all_hidden = None
    all_hidden = []
    all_label = []
    last = 0
    for ii, batch in zip(it, data_train):
      if rgb:
        batch['x'] = tf.concat(tf.split(batch['x'], 3, axis=-1)[::-1], axis=-1)
      h = model.backbone(batch['x'], is_training=is_training)
      B = int(h.shape[0])
      # if all_hidden is None:
      #   all_hidden = np.zeros([len(data_train) * B, h.shape[1]])
      # all_hidden[last:last + B] = h.numpy()
      all_hidden.append(h.numpy())
      # last += B
      all_label.append(batch['y'].numpy())
      # if ii > 10:
      #   break
    all_hidden = np.concatenate(all_hidden, axis=0)
    print(all_hidden.shape)

    it_val = range(len(data_val))
    it_val = tqdm(it_val, ncols=0, desc='val')
    all_val_hidden = []
    all_val_label = []
    # last = 0
    for ii, batch in zip(it_val, data_val):
      if rgb:
        batch['x'] = tf.concat(tf.split(batch['x'], 3, axis=-1)[::-1], axis=-1)
      h = model.backbone(batch['x'], is_training=is_training)
      # B = int(h.shape[0])
      # if all_val_hidden is None:
      #   all_val_hidden = np.zeros([len(data_val) * B, h.shape[1]])
      all_val_hidden.append(h.numpy())
      # last += B
      all_val_label.append(batch['y'].numpy())
      # if ii > 10:
      #   break
    all_val_hidden = np.concatenate(all_val_hidden, axis=0)
    print(all_val_hidden.shape)

    it_test = range(len(data_test))
    it_test = tqdm(it_test, ncols=0, desc='test')
    all_test_hidden = []
    all_test_label = []
    # last = 0
    for ii, batch in zip(it_test, data_test):
      if rgb:
        batch['x'] = tf.concat(tf.split(batch['x'], 3, axis=-1)[::-1], axis=-1)
      h = model.backbone(batch['x'], is_training=is_training)
      # B = int(h.shape[0])
      # if all_test_hidden is None:
      #   all_test_hidden = np.zeros([len(data_test) * B, h.shape[1]])
      all_test_hidden.append(h.numpy())
      # last += B
      all_test_label.append(batch['y'].numpy())
      # if ii > 10:
      #   break
    all_test_hidden = np.concatenate(all_test_hidden, axis=0)
    print(all_test_hidden.shape)

    with tf.device('/cpu:0'):
      all_hidden = tf.linalg.l2_normalize(all_hidden, axis=-1)
      all_val_hidden = tf.linalg.l2_normalize(all_val_hidden, axis=-1)
      all_test_hidden = tf.linalg.l2_normalize(all_test_hidden, axis=-1)
      all_label = tf.concat(all_label, axis=0)
      all_val_label = tf.concat(all_val_label, axis=0)
      all_test_label = tf.concat(all_test_label, axis=0)
      sim_val = tf.matmul(all_val_hidden, all_hidden, transpose_b=True)
      sim_test = tf.matmul(all_test_hidden, all_hidden, transpose_b=True)
      best_k = 0
      best_val_acc = 0.0
      for k in range(1, 41, 2):
        all_pred = []
        for i in range(all_val_hidden.shape[0]):
          results = tf.math.top_k(sim_val[i], k)
          idx = results.indices
          top_k = tf.gather(all_label, idx)  # [K]
          # print(top_k.shape)
          y, _, count = tf.unique_with_counts(top_k)
          major = tf.argmax(count)
          all_pred.append(y[major])
        all_pred = tf.stack(all_pred).numpy()
        acc = np.equal(all_pred, all_val_label).astype(np.float32).mean()
        if acc > best_val_acc:
          best_k = k
          best_val_acc = acc
        print('k', k, 'acc', '{:.3f}'.format(acc * 100.0))

      print('select k={}'.format(best_k))
      k = best_k
      all_pred = []
      for i in range(all_test_hidden.shape[0]):
        results = tf.math.top_k(sim_test[i], k)
        idx = results.indices
        top_k = tf.gather(all_label, idx)  # [K]
        y, _, count = tf.unique_with_counts(top_k)
        major = tf.argmax(count)
        all_pred.append(y[major])
      all_pred = tf.stack(all_pred).numpy()
      acc = np.equal(all_pred, all_test_label).astype(np.float32).mean()
      print('test acc', '{:.3f}'.format(acc * 100.0))

  return _eval_fn


def get_evaluate_fewshot_fn(model,
                            data_train,
                            data_val,
                            data_test,
                            logger,
                            nepisode=120,
                            verbose=False):

  def _eval_fn(step):
    # print(model)
    if data_train is not None:
      trainfs_results = evaluate_fs(
          model, data_train, nepisode, verbose=verbose)
      # print('Step {:d} train FS readout: {:.3f}'.format(
      #     step, trainfs_results['acc'] * 100.0))
    if data_val is not None:
      valfs_results = evaluate_fs(model, data_val, nepisode, verbose=verbose)
      # print('Step {:d} val FS readout: {:.3f}'.format(
      #     step, valfs_results['acc'] * 100.0))
    if data_test is not None:
      testfs_results = evaluate_fs(model, data_test, nepisode, verbose=verbose)
      # print('Step {:d} test FS readout: {:.3f}'.format(
      #     step, testfs_results['acc'] * 100.0))
    if step > 0 and logger is not None:
      if data_train is not None:
        logger.log('fs acc train', step, trainfs_results['acc'] * 100.0)
      if data_val is not None:
        logger.log('fs acc val', step, valfs_results['acc'] * 100.0)
      if data_test is not None:
        logger.log('fs acc test', step, testfs_results['acc'] * 100.0)
      logger.flush()
    if data_val is not None:
      return valfs_results['acc']
    else:
      return None

  return _eval_fn


def get_evaluate_readout_fn(model,
                            data_train,
                            data_val,
                            logger,
                            nepoch=10,
                            is_training=False,
                            acc_fn=top1_acc,
                            rgb=False,
                            is_chief=False,
                            prefix=''):
  """Readout classification.

  Args:
    model: Network.
    data_train: Training data.
    data_val: Test data.
    logger: TensorBorad logger, no log output if None.
    is_training: Bool. Whether to turn on training mode during readout.
  """

  def _eval_fn(step):
    # model.backbone.set_trainable(False)

    # Reinitialize.
    model._fc._weight.assign(tf.zeros_like(model._fc._weight))
    model._fc._bias.assign(tf.zeros_like(model._fc._bias))
    # print("Optimizer weights 0", model.optimizer.get_weights())
    # assert False
    # assert False

    for e in range(0, nepoch):
      data_train.reset()
      it = range(len(data_train))
      if is_chief:
        it = tqdm(it, ncols=0, desc="epoch {}".format(e))
      for ii, batch in zip(it, data_train):
        if rgb:
          batch['x'] = tf.concat(
              tf.split(batch['x'], 3, axis=-1)[::-1], axis=-1)
        loss = model.train_step(
            batch['x'], batch['y'], is_training=is_training)

        if is_chief:
          postfix_dict = {}
          postfix_dict['loss'] = '{:.3f}'.format(loss)
          it.set_postfix(**postfix_dict)

      acc_list = []
      data_val.reset()
      itval = range(len(data_val))
      for ii, batch in zip(itval, data_val):
        if rgb:
          batch['x'] = tf.concat(
              tf.split(batch['x'], 3, axis=-1)[::-1], axis=-1)
        pred = model.eval_step(batch['x'])
        pred_all = hvd.allgather(pred)
        y_all = hvd.allgather(batch['y'])
        # acc_list.append(acc_fn(pred, batch['y'].numpy()))
        acc_list.append(acc_fn(pred_all.numpy(), y_all.numpy()))
      acc = np.array(acc_list).mean()
      # print(model.learn_rate)
      # print(model._optimizer)
      # print(model._fc._weight)

      if is_chief:
        print('Epoch {} readout acc {:.3f}'.format(e, acc * 100.0))

    if is_chief:
      if logger is not None:
        logger.log(prefix + 'readout val acc', step, acc * 100.0)
        logger.flush()

    # model.backbone.set_trainable(True)
    opt_weights = model.optimizer.get_weights()
    print("Optimizer weights", model.optimizer.get_weights())

    # Clear momentums.
    model.optimizer.set_weights([np.zeros_like(v) for v in opt_weights])
    # print("Optimizer weights 2", model.optimizer.get_weights())
    return acc

  return _eval_fn


def get_evaluate_readout_v2_fn(model,
                               data_train,
                               data_val,
                               logger,
                               nepoch=10,
                               add_relu=False,
                               is_training=False,
                               acc_fn=top1_acc,
                               rgb=False,
                               is_chief=False,
                               prefix=''):
  """Readout classification.

  Args:
    model: Network.
    data_train: Training data.
    data_val: Test data.
    logger: TensorBorad logger, no log output if None.
    is_training: Bool. Whether to turn on training mode during readout.
  """

  from sklearn import preprocessing
  from sklearn.linear_model import LogisticRegression

  def _eval_fn(step):
    model._fc._weight.assign(tf.zeros_like(model._fc._weight))
    model._fc._bias.assign(tf.zeros_like(model._fc._bias))

    data_train.reset()
    it = range(len(data_train))
    X_train = []
    y_train = []
    for ii, batch in zip(it, data_train):
      if rgb:
        batch['x'] = tf.concat(tf.split(batch['x'], 3, axis=-1)[::-1], axis=-1)
      h = model._backbone(batch['x'], is_training=is_training)
      X_train.append(h.numpy())
      y_train.append(batch['y'].numpy())

    data_val.reset()
    X_test = []
    y_test = []
    for ii, batch in zip(it, data_val):
      if rgb:
        batch['x'] = tf.concat(tf.split(batch['x'], 3, axis=-1)[::-1], axis=-1)
      h = model._backbone(batch['x'], is_training=is_training)
      X_test.append(h.numpy())
      y_test.append(batch['y'].numpy())
    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    if add_relu:
      X_train = np.maximum(X_train, 0.0)
      X_test = np.maximum(X_test, 0.0)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    lr_model = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=nepoch)

    lr_model.fit(X_train, y_train)
    y_hat_train = lr_model.predict(X_train)
    y_hat_test = lr_model.predict(X_test)
    train_acc = (y_hat_train == y_train).astype(np.float32).mean()
    test_acc = (y_hat_test == y_test).astype(np.float32).mean()
    print('Train acc {:.3f} Test acc {:.3f}'.format(train_acc * 100.0,
                                                    test_acc * 100.0))
    acc = test_acc

    if is_chief:
      if logger is not None:
        logger.log(prefix + 'readout val acc', step, acc)
        logger.log(prefix + 'readout train acc', step, train_acc)
        logger.flush()
    return acc

  return _eval_fn


def train(model,
          dataiter,
          dataiter_test,
          eval_fn,
          ckpt_folder,
          final_save_folder,
          logger,
          writer,
          is_chief,
          reload_flag,
          save_all=False):

  Nsteps = len(dataiter)
  Nepoch = model.config.optimizer_config.max_train_epochs
  N = Nepoch * Nsteps
  config = model.config.train_config
  start = model.step.numpy()
  if start > 0:
    log.info('Restore from step {}'.format(start))
  estart = start // Nsteps
  sstart = start % Nsteps
  kwargs = {}
  kwargs['writer'] = writer
  acc = None

  for e in range(estart, Nepoch):
    dataiter.reset()
    if e == estart:
      it = six.moves.xrange(sstart, Nsteps)
    else:
      it = six.moves.xrange(Nsteps)
    if is_chief:
      it = tqdm(it, ncols=0)

    step = model.step.numpy()
    for ii, batch in zip(it, dataiter):
      # if 'id' in batch:
      #   print('rank', hvd.rank(), 'data id', batch['id'], batch['x'].shape)
      loss = model.train_step(batch['x'], **kwargs)
      step = model.step.numpy()

      if step % config.steps_per_save == 0 and is_chief:
        if model.backbone.config.data_format == "NCHW":
          x1, x2 = tf.split(batch['x'], 2, axis=1)
          x1 = tf.transpose(x1, [0, 2, 3, 1])
          x2 = tf.transpose(x2, [0, 2, 3, 1])
        else:
          x1, x2 = tf.split(batch['x'], 2, axis=-1)
        tf.summary.image('inp/x1', x1, step=step, max_outputs=3)
        tf.summary.image('inp/x2', x2, step=step, max_outputs=3)

      # Reload model weights.
      if ii == sstart and e == estart and reload_flag is not None:
        model.load(reload_flag, load_optimizer=True)

      # Horovod init.
      if ii == sstart and e == estart and model._distributed:
        hvd.broadcast_variables(model.var_to_optimize(), root_rank=0)
        hvd.broadcast_variables(model.optimizer.variables(), root_rank=0)
        if model.config.set_backbone_lr:
          hvd.broadcast_variables(model._bb_optimizer.variables(), root_rank=0)

      # Save.
      # print(step)
      if (step % config.steps_per_save == 0) and is_chief:
        cur_fname = os.path.join(ckpt_folder, 'weights-{}.pkl'.format(step))
        model.save(cur_fname)
        # print(cur_fname)
        cur_fname = os.path.join(final_save_folder,
                                 'weights-{}.pkl'.format(step))
        model.save(cur_fname)

        # Delete earlier checkpoint.
        if not save_all and ckpt_folder == final_save_folder:
          delete_old_ckpt(ckpt_folder, "weights-", step + 1,
                          1 * config.steps_per_save)
        else:
          fnames = glob.glob(os.path.join(final_save_folder, 'weights-*'))
          for f in fnames:
            name = f.split('-')[-1].split('.')[0]
            if step - int(name) >= config.steps_per_save:
              os.remove(f)

      # Evaluation.
      is_val = step % config.steps_per_val == 0
      # if (is_val or step == 1) and is_chief and eval_fn is not None:
      if (is_val or step == 1) and eval_fn is not None:
        # if is_val and eval_fn is not None:
        acc = eval_fn(step)

      # Write logs.
      if step % config.steps_per_log == 0 and is_chief:
        lr = model.learn_rate().numpy()

        # Update progress bar.
        logger.log('loss', step, loss)
        logger.log('lr', step, lr)
        logger.flush()
        post_fix_dict = {}
        post_fix_dict['epoch'] = '{:d}'.format(e)
        post_fix_dict['lr'] = '{:.3e}'.format(lr)
        post_fix_dict['steps'] = '{:d}'.format(step)
        post_fix_dict['loss'] = '{:.3e}'.format(loss)
        if acc is not None:
          post_fix_dict['acc'] = '{:.3f}'.format(acc * 100.0)

        it.set_postfix(**post_fix_dict)

  # Save.
  if is_chief and final_save_folder is not None:
    model.save(os.path.join(final_save_folder, 'weights-{}.pkl'.format(N)))


def main():
  assert tf.executing_eagerly(), 'Only eager mode is supported.'
  assert args.config is not None, 'You need to pass in model config file path'
  assert args.env is not None, 'You need to pass in environ config file path'
  assert args.tag is not None, 'You need to specify a tag'

  config = get_config(args.config, ExperimentConfig)
  env_config = get_config(args.env, EnvironmentConfig)
  dataset = get_data(env_config)
  if args.data is not None:
    non_iid_data_config = get_config(args.data, EpisodeConfig)
  else:
    non_iid_data_config = None
  if args.readout_env is not None:
    readout_env_config = get_config(args.readout_env, EnvironmentConfig)
  else:
    readout_env_config = env_config
  reload_flag = None
  restore_steps = 0
  distributed = hvd.size() > 1
  num_train_examples = dataset['train'].get_size()

  if args.vid:
    num_train_examples /= non_iid_data_config.frame_rate

  model = build_contrastive_net(
      config, num_train_examples=num_train_examples, distributed=distributed)
  reload_flag = None

  if args.area_lb > 0.0:
    log.info('Area lb modified to {:.3f}'.format(args.area_lb))
    env_config.area_range_lb = args.area_lb

  if is_chief:
    # Load pretrain checkpoint.
    if args.pretrain:
      latest_pretrain = latest_file(args.pretrain, 'weights-')
      log.info('Loading pretrain weights from {}'.format(args.pretrain))
      model.load(latest_pretrain)

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
  else:
    save_folder = None
    ckpt_folder = None
    writer = None
    logger = None

  # assert hvd.size() > 1, 'not distributed'
  # print(hvd.size(), 'total size')
  if args.max_len > 0:
    non_iid_data_config.maxlen = args.max_len
  data = get_dataiter_contrastive(
      dataset,
      config.optimizer_config.batch_size,
      nchw=model.backbone.config.data_format == 'NCHW',
      distributed=hvd.size() > 1,
      seed=args.seed + restore_steps,
      color_distort=env_config.random_color,
      flip=env_config.random_flip,
      color_distort_strength=env_config.random_color_strength,
      area_range_lb=env_config.area_range_lb,
      sim=args.sim,
      vid=args.vid,
      background_ratio=args.background_ratio,
      only_one=args.only_one,
      non_iid=args.non_iid,
      non_iid_data_config=non_iid_data_config)

  with writer.as_default() if writer is not None else dummy_context_mgr(
  ) as gs:

    eval_fn = None
    if args.eval_type == 'fewshot':
      config.model_class = "proto_net"  # TODO change this to classifier_net
      config.optimizer_config.optimizer = "adam"
      eval_data_config = get_config(args.eval_data_config, EpisodeConfig)
      proto_net = build_fewshot_net(config, backbone=model.backbone)
      dataset_fs = get_data_fs(
          readout_env_config,
          load_train=False,
          load_val=is_chief,
          load_test=False)
      dataset_fs['train_fs'] = dataset['train']
      data_fs = get_dataiter_fewshot(
          dataset_fs,
          # dataset,
          eval_data_config,
          nchw=model.backbone.config.data_format == 'NCHW')
      eval_fn = get_evaluate_fewshot_fn(proto_net, data_fs['train_fs'],
                                        data_fs['val_fs'], None, logger)
    elif args.eval_type == 'readout':
      config.model_class = "pretrain_net"
      config.optimizer_config.optimizer = "adam"
      readout_net = build_pretrain_net(config, backbone=model.backbone)
      readout_net._learn_rate = tf.constant(1e-3)
      readout_net._optimizer = readout_net._get_optimizer(
          "adam", readout_net.learn_rate)
      # Only optimize the last layer.
      readout_net._var_to_optimize = [
          readout_net._fc._weight, readout_net._fc._bias
      ]
      if args.readout_env is not None:
        readout_dataset = get_data(readout_env_config)
      else:
        readout_dataset = dataset
      readout_data = get_dataiter_epoch(
          readout_dataset,
          256,
          nchw=model.backbone.config.data_format == 'NCHW',
          distributed=distributed)
      if args.readout_v2:
        eval_fn = get_evaluate_readout_v2_fn(
            readout_net,
            readout_data['train'],
            readout_data['val'],
            logger,
            nepoch=100,
            add_relu=args.eval_with_relu,
            is_training=False,
            is_chief=is_chief)
      else:
        eval_fn = get_evaluate_readout_fn(
            readout_net,
            readout_data['train'],
            readout_data['val'],
            logger,
            is_training=True,
            is_chief=is_chief)

    if not args.eval:
      train(
          model,
          data['train'],
          None,
          eval_fn,
          ckpt_folder,
          save_folder,
          logger,
          writer,
          is_chief,
          reload_flag=reload_flag)

    elif args.eval_type is not None:
      eval_fn(-1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Pretrain Contrastive")
  parser.add_argument('--config', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--data', type=str, default=None)
  parser.add_argument('--tag', type=str, default=None)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--pretrain', type=str, default=None)

  # To see few-shot performance in the middle of the training.
  # "fewshot" or "readout"
  parser.add_argument('--eval_type', type=str, default=None)

  # If the readout dataset is different from the training dataset.
  parser.add_argument('--readout_env', type=str, default=None)

  # Use scipy to readout.
  parser.add_argument('--readout_v2', action='store_true')
  parser.add_argument('--eval_with_relu', action='store_true')

  # Only used for few-shot, to define support/query.
  parser.add_argument('--eval_data_config', type=str, default=None)
  parser.add_argument('--area_lb', type=float, default=-1.0)
  parser.add_argument('--background_ratio', type=float, default=-1.0)
  parser.add_argument('--only_one', action='store_true')
  parser.add_argument('--sim', action='store_true')
  parser.add_argument('--non_iid', action='store_true')
  parser.add_argument('--max_len', type=int, default=-1)
  parser.add_argument('--vid', action='store_true')

  args = parser.parse_args()
  main()
