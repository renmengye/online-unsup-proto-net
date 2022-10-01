"""
Evaluate the pretrained network using readout/fewshot.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
  eval_contrastive.py --config [CONFIG] --tag [TAG} --dataset [DATASET] \
              --data_folder [DATA FOLDER] --results [SAVE FOLDER] \
              --eval_type [EVAL TYPE] [--eval_data_config]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
import glob
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

from fewshot.configs.environment_config_pb2 import EnvironmentConfig
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.experiments.build_model import build_contrastive_net
from fewshot.experiments.build_model import build_fewshot_net
from fewshot.experiments.build_model import build_pretrain_net
from fewshot.experiments.get_data_iter import get_dataiter_fewshot, get_dataiter_matterport_category
from fewshot.experiments.get_data_iter import get_dataiter_epoch
from fewshot.experiments.utils import ExperimentLogger
from fewshot.experiments.utils import get_config
from fewshot.experiments.utils import get_data
from fewshot.experiments.utils import get_data_fs
from fewshot.experiments.utils import save_config
from fewshot.experiments.pretrain_contrastive import get_evaluate_readout_fn
from fewshot.experiments.pretrain_contrastive import get_evaluate_readout_v2_fn
from fewshot.experiments.pretrain_contrastive import get_evaluate_nn_fn
from fewshot.experiments.pretrain_contrastive import get_evaluate_fewshot_fn
# from fewshot.experiments.pretrain_binary import binary_acc
from fewshot.utils.dummy_context_mgr import dummy_context_mgr
from fewshot.utils.logger import get as get_logger

log = get_logger()


def main():
  assert tf.executing_eagerly(), 'Only eager mode is supported.'
  assert args.config is not None, 'You need to pass in model config file path'
  assert args.env is not None, 'You need to pass in environ config file path'

  config = get_config(args.config, ExperimentConfig)
  env_config = get_config(args.env, EnvironmentConfig)
  dataset = get_data(env_config)
  distributed = hvd.size() > 1
  model = build_contrastive_net(
      config,
      num_train_examples=dataset['train'].get_size(),
      distributed=distributed)

  if args.tag is not None:
    save_folder = os.path.join(env_config.results, env_config.dataset,
                               args.tag)
  elif args.reload is not None:
    save_folder = args.reload
  else:
    assert False, 'You need to specify a reload folder'
  if not args.random:
    if args.usebest:
      prefix = "best-"
    else:
      prefix = "weights-"
    list_of_files = glob.glob(os.path.join(save_folder, prefix + '*'))
    if len(list_of_files) == 0:
      return None

    def sort_fn(f):
      name = f.split('-')[-1].split('.')[0]
      if name == 'final':
        return 100000000
      else:
        return int(name)

    list_of_files = sorted(list_of_files, key=sort_fn)
    # print(list_of_files)
    # assert False

  if args.write_log:
    writer = tf.summary.create_file_writer(save_folder)
    logger = ExperimentLogger(writer)
  else:
    writer = None
    logger = None

  with writer.as_default() if writer is not None else dummy_context_mgr(
  ) as gs:
    eval_fn = None
    if args.eval_type == 'fewshot':
      eval_model_config = get_config(args.eval_model_config, ExperimentConfig)
      eval_data_config = get_config(args.eval_data_config, EpisodeConfig)
      net = build_fewshot_net(eval_model_config, backbone=model.backbone)
      dataset_fs = get_data_fs(env_config, load_train=True)
      data_fs = get_dataiter_fewshot(
          dataset_fs,
          eval_data_config,
          nchw=model.backbone.config.data_format == 'NCHW',
          data_aug=eval_data_config.data_aug)
      train_data = data_fs['train_fs_eval']
      val_data = data_fs['val_fs']
      if args.test:
        test_data = data_fs['test_fs']
      else:
        test_data = None
      eval_fn = get_evaluate_fewshot_fn(
          net,
          train_data,
          val_data,
          test_data,
          logger,
          nepisode=args.nepisode,
          verbose=True)
    elif args.eval_type in ['readout', 'readout_v2']:
      config.model_class = "pretrain_net"
      config.optimizer_config.optimizer = "adam"  # Dummy.
      config.resnet_config.weight_decay = 0e0
      net = build_pretrain_net(config, backbone=model.backbone)
      log.info('Readout SGD learning rate {:.3f}'.format(args.lr))
      net._learn_rate = tf.constant(args.lr)
      # Only optimize the last layer.
      net._var_to_optimize = [net._fc._weight, net._fc._bias]
      net._optimizer = net._get_optimizer(args.optimizer, net.learn_rate)
      readout_dataset = get_data(env_config)

      if env_config.dataset in ['matterport', 'roaming-rooms']:
        for d in readout_dataset.keys():
          if d != 'metadata':
            readout_dataset[d]._train_size = 128000  # Smaller.
        readout_data = get_dataiter_matterport_category(
            readout_dataset,
            128,
            nchw=model.backbone.config.data_format == 'NCHW',
            distributed=distributed)
      else:
        readout_data = get_dataiter_epoch(
            readout_dataset,
            256,
            nchw=model.backbone.config.data_format == 'NCHW',
            distributed=distributed)
      train_data = readout_data['train']
      val_data = readout_data['val']

      # For random readout, need to turn on is_training.
      if args.eval_type == 'readout':
        eval_fn = get_evaluate_readout_fn(
            net,
            train_data,
            val_data,
            logger,
            nepoch=args.nepoch,
            rgb=args.rgb,
            is_training=args.random,
            is_chief=is_chief,
            prefix=args.prefix)
      elif args.eval_type == 'readout_v2':
        eval_fn = get_evaluate_readout_v2_fn(
            net,
            train_data,
            val_data,
            logger,
            nepoch=args.nepoch,
            rgb=args.rgb,
            is_training=args.random,
            is_chief=is_chief,
            prefix=args.prefix)
    elif args.eval_type == 'nearest_neighbor':
      net = build_pretrain_net(config, backbone=model.backbone)
      readout_dataset = get_data(env_config)
      if env_config.dataset in ['matterport', 'roaming-rooms']:
        for d in readout_dataset.keys():
          if d != 'metadata':
            readout_dataset[d]._train_size = 128000  # Smaller.
        readout_data = get_dataiter_matterport_category(
            readout_dataset,
            128,
            nchw=model.backbone.config.data_format == 'NCHW')
      else:
        readout_data = get_dataiter_epoch(
            readout_dataset,
            256,
            nchw=model.backbone.config.data_format == 'NCHW')
      train_data = readout_data['train']
      val_data = readout_data['val']
      test_data = readout_data['test']
      eval_fn = get_evaluate_nn_fn(
          net, train_data, val_data, test_data, is_training=args.random)
      pass
    # elif args.eval_type == 'readout_binary':
    #   # Binary attribute readout. Made for Celeb-A.
    #   config.model_class = "pretrain_sigmoid_net"
    #   config.optimizer_config.optimizer = "adam"  # Dummy.
    #   config.resnet_config.weight_decay = 0e0
    #   if args.all_label:
    #     # Feed in the true attributes.
    #     kwargs = {"all_label": True}
    #     config.num_classes = 40
    #   else:
    #     kwargs = {}
    #     config.num_classes = 14
    #     assert False
    #   net = build_pretrain_net(config, backbone=model.backbone)
    #   net._learn_rate = tf.constant(1e-3)
    #   net._optimizer = net._get_optimizer("adam", net.learn_rate)
    #   readout_dataset = get_data(env_config, **kwargs)
    #   readout_data = get_dataiter_epoch(
    #       readout_dataset,
    #       256,
    #       nchw=model.backbone.config.data_format == 'NCHW')
    #   train_data = readout_data['train']
    #   val_data = readout_data['val']
    #   eval_fn = get_evaluate_readout_fn(
    #       net,
    #       train_data,
    #       val_data,
    #       logger,
    #       nepoch=args.nepoch,
    #       is_training=args.random,
    #       acc_fn=binary_acc)
    else:
      assert False, "Eval type not supported"

    # Run evaluation on each model checkpoint.
    if args.last and not args.random:
      list_of_files = [list_of_files[-1]]
      # print(list_of_files)
    with log.verbose_level(0):
      if args.random:
        step = 0
        eval_fn(step)
      else:
        for f in list_of_files:
          print(f)
          model.load(f)
          name = f.split('-')[-1].split('.')[0]
          if name != 'final':
            step = int(name)
            if step < args.start_checkpoint:
              continue
          eval_fn(step)

      if args.eval_type in ['readout', 'readout_binary']:
        # save_config(config, save_folder, name='readout.prototxt')
        # net.save(os.path.join(save_folder, 'readout-{}.pkl'.format(step)))
        net.save(os.path.join(save_folder, 'readout-v2-{}.pkl'.format(step)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Pretrain Contrastive")
  parser.add_argument('--config', type=str, default=None)
  parser.add_argument('--env', type=str, default=None)
  parser.add_argument('--tag', type=str, default=None)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--eval', action='store_true')

  # To see few-shot performance in the middle of the training.
  # "fewshot" or "readout" or "readout_binary"
  parser.add_argument('--eval_type', type=str, default=None)

  # Only used for few-shot, to define support/query.
  parser.add_argument('--eval_data_config', type=str, default=None)
  parser.add_argument('--eval_model_config', type=str, default=None)

  # Whether to only evaluate the last checkpoint.
  parser.add_argument('--last', action='store_true')

  # Number of readout epochs.
  parser.add_argument('--nepoch', type=int, default=10)

  # Number of fewshot episodes.
  parser.add_argument('--nepisode', type=int, default=120)

  # # For CIFAR, train=trainval, val=test.
  # parser.add_argument('--train_split', type=str, default="train")
  # parser.add_argument('--val_split', type=str, default="val")
  parser.add_argument('--start_checkpoint', type=int, default=0)

  parser.add_argument('--test', action='store_true')

  # Whether to read out all attributes.
  parser.add_argument('--all_label', action='store_true')

  # Whether to use a random initialization.
  parser.add_argument('--random', action='store_true')

  parser.add_argument('--write_log', action='store_true')

  parser.add_argument("--usebest", action="store_true")
  # Whether to use RGB format (for imagenet initialization)
  parser.add_argument("--rgb", action="store_true")

  # Special data iterator for matterport.
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--optimizer', type=str, default="sgd")

  parser.add_argument('--reload', type=str, default=None)
  parser.add_argument('--prefix', type=str, default="aftereval/")
  # Number of neighbors, for nearest neighbor readout.
  # parser.add_argument('--knn', type=int, default=1)
  args = parser.parse_args()
  main()
