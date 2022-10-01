"""
Build data iterators.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from fewshot.configs.episode_config_pb2 import EpisodeConfig
from fewshot.data.iterators.matterport_category_iterator import MatterportCategoryIterator  # NOQA
from fewshot.data.iterators.mix_unsupervised_augment_sim_episode_iterator import MixUnsupAugmentSimEpisodeIterator  # NOQA
from fewshot.data.iterators.mix_unsupervised_augment_sim_iterator_noniid_v2 import MixUnsupAugmentSimIteratorNonIIDV2  # NOQA
from fewshot.data.iterators.mix_unsupervised_augment_vid_iterator import MixUnsupAugmentVidIterator  # NOQA
from fewshot.data.iterators.mix_unsupervised_augment_vid_iterator_noniid import MixUnsupAugmentVidIteratorNonIID  # NOQA
from fewshot.data.iterators.unsupervised_augment_episode_iterator_noniid import UnsupervisedAugmentEpisodeIteratorNonIID  # NOQA
from fewshot.data.iterators.epoch_file_minibatch_iterator import EpochFileMinibatchIterator  # NOQA
from fewshot.data.iterators.unsupervised_augment_sim_episode_iterator_multigpu import UnsupervisedAugmentSimEpisodeIteratorMultiGPU  # NOQA
from fewshot.data.iterators.unsupervised_augment_vid_episode_iterator import UnsupervisedAugmentVidEpisodeIterator  # NOQA

from fewshot.data.iterators.mix_unsupervised_augment_sim_iterator import MixUnsupAugmentSimIterator  # NOQA
from fewshot.data.iterators.mix_unsupervised_augment_sim_iterator_noniid import MixUnsupAugmentSimIteratorNonIID  # NOQA
import numpy as np

from fewshot.data.data_factory import get_sampler
from fewshot.data.iterators import EpisodeIterator
from fewshot.data.iterators import MinibatchIterator
from fewshot.data.iterators import EpochMinibatchIteratorV2
from fewshot.data.iterators import SIMCLRMinibatchIterator
from fewshot.data.iterators import SIMCLRImbalanceMinibatchIterator
from fewshot.data.iterators import SIMCLRAugMinibatchIterator
from fewshot.data.iterators import SemiSupervisedEpisodeIterator
from fewshot.data.iterators import UnsupervisedEpisodeIterator
from fewshot.data.iterators import UnsupervisedAugmentEpisodeIterator
from fewshot.data.iterators import UnsupervisedAugmentImbalanceEpisodeIterator
from fewshot.data.iterators import UnsupervisedAugmentSimEpisodeIterator
from fewshot.data.iterators import SimEpisodeIterator
from fewshot.data.preprocessors import DataAugmentationPreprocessor
from fewshot.data.preprocessors import NCHWPreprocessor
from fewshot.data.preprocessors import SequentialPreprocessor
from fewshot.data.preprocessors import RandomBoxOccluder
from fewshot.data.preprocessors import FloatPreprocessor
from fewshot.data.samplers import FewshotSampler
from fewshot.data.samplers import HierarchicalEpisodeSampler
from fewshot.data.samplers import MinibatchSampler
from fewshot.data.samplers import ContinualMinibatchSampler
from fewshot.data.samplers import MixSampler
from fewshot.data.samplers import SemiSupervisedEpisodeSampler
from fewshot.data.samplers import UnsupervisedEpisodeSampler
from fewshot.data.samplers import ContinualEpisodeSampler
from fewshot.data.datasets import UppsalaDataset
from fewshot.data.episode_processors import BackgroundProcessor
from fewshot.data.episode_processors import BackgroundProcessorV2
from fewshot.data.samplers.blender import get_blender


def get_dataiter(data,
                 batch_size,
                 nchw=False,
                 data_aug=False,
                 distributed=False,
                 continual=False,
                 stage=None):
  """Gets dataset iterator."""
  md = data['metadata']
  if distributed:
    import horovod.tensorflow as hvd
    rank = hvd.rank()
    seed = rank * 1234
  else:
    seed = 0
  sampler_dict = {}
  if not continual:
    sampler_dict['train'] = MinibatchSampler(seed, cycle=True, shuffle=True)
    sampler_dict['val'] = MinibatchSampler(seed, cycle=False, shuffle=False)
    sampler_dict['test'] = MinibatchSampler(seed, cycle=False, shuffle=False)
    split_list = ['train', 'val', 'test']
  else:
    sampler_dict['train'] = ContinualMinibatchSampler(seed)
    assert stage is not None
    sampler_dict['train'].set_stage(stage)
    split_list = ['train']

  # norm_prep = NormalizationPreprocessor(
  #     mean=np.array(md.mean_pix), std=np.array(md.std_pix))
  norm_prep = FloatPreprocessor()
  da_prep = DataAugmentationPreprocessor(md.image_size, md.crop_size,
                                         md.random_crop, md.random_flip,
                                         md.random_color, md.random_rotate)
  if nchw:
    nchw_prep = NCHWPreprocessor()
    if data_aug:
      train_prep = SequentialPreprocessor(da_prep, norm_prep, nchw_prep)
    else:
      train_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    val_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    test_prep = SequentialPreprocessor(norm_prep, nchw_prep)
  else:
    train_prep = SequentialPreprocessor(da_prep, norm_prep)
    val_prep = norm_prep
    test_prep = norm_prep
  prep = {'train': train_prep, 'val': val_prep, 'test': test_prep}
  it_dict = {}
  for k in split_list:
    cycle = k == 'train'
    if k in data:
      it_dict[k] = MinibatchIterator(
          data[k],
          sampler_dict[k],
          batch_size,
          prefetch=True,
          preprocessor=prep[k])
  return it_dict


def get_dataiter_fewshot(data,
                         data_config,
                         batch_size=1,
                         nchw=False,
                         prefetch=True,
                         distributed=False):
  """Gets few-shot episode iterator."""
  md = data['metadata']
  if distributed:
    import horovod.tensorflow as hvd
    rank = hvd.rank()
    seed = rank * 1234
  else:
    seed = 0
  sampler_dict = {}
  sampler_dict['train_fs'] = FewshotSampler(seed)
  sampler_dict['val_fs'] = FewshotSampler(seed)
  sampler_dict['test_fs'] = FewshotSampler(seed)
  # norm_prep = NormalizationPreprocessor(
  #     mean=np.array(md.mean_pix), std=np.array(md.std_pix))
  norm_prep = FloatPreprocessor()
  if nchw:
    nchw_prep = NCHWPreprocessor()
    prep = SequentialPreprocessor(norm_prep, nchw_prep)
  else:
    prep = norm_prep
  it_dict = {}
  # For evaluation only. No additional preprocessor.
  for k in ['train_fs', 'val_fs', 'test_fs']:
    if data[k] is not None:
      it_dict[k] = EpisodeIterator(
          data[k],
          data_config,
          sampler_dict[k],
          batch_size=batch_size,
          preprocessor=prep)
    else:
      it_dict[k] = None
  return it_dict


def get_dataiter_continual(data,
                           data_config,
                           batch_size=1,
                           nchw=True,
                           prefetch=True,
                           save_additional_info=False,
                           random_box=False,
                           distributed=False,
                           stage=None,
                           seed=0):
  """Gets few-shot episode iterator.

  Args:
    data: Object. Dataset object.
    data_config: Config. Data source episode config.
    batch_size: Int. Batch size.
    nchw: Bool. Whether to transpose the images to NCHW.
    prefetch: Bool. Whether to add prefetching module in the data loader.
    save_additional_info: Bool. Whether to add additional episodic information.
  """
  md = data['metadata']
  data['trainval_fs'] = data['train_fs']  # Use the same for trainval.
  if data_config.base_sampler == 'incremental':
    kwargs = {
        'nshot_min': 1,
        'nshot_max': data_config.nshot_max,
        'allow_repeat': data_config.allow_repeat
    }
  elif data_config.base_sampler == 'fewshot':
    kwargs = {
        'nshot_min': data_config.nshot_max,
        'nshot_max': data_config.nshot_max,
        'allow_repeat': data_config.allow_repeat
    }
  elif data_config.base_sampler == 'constant_prob':
    kwargs = {
        'p': data_config.prob_new,
        'allow_repeat': data_config.allow_repeat,
        'max_num': data_config.maxlen,
        'max_num_per_cls': data_config.max_num_per_cls
    }
  elif data_config.base_sampler == 'crp':
    kwargs = {
        'alpha': data_config.crp_alpha,
        'theta': data_config.crp_theta,
        'allow_repeat': data_config.allow_repeat,
        'max_num': data_config.maxlen,
        'max_num_per_cls': data_config.max_num_per_cls
    }
  elif data_config.base_sampler == 'seq_crp':
    kwargs = {
        'alpha': data_config.crp_alpha,
        'theta': data_config.crp_theta,
        'allow_repeat': data_config.allow_repeat,
        'max_num': data_config.maxlen // 2,
        'max_num_per_cls': data_config.max_num_per_cls,
        'stages': 2
    }
  else:
    raise ValueError('Not supported')

  # Split.
  split_list = ['train_fs', 'trainval_fs', 'val_fs', 'test_fs']
  sampler_dict = {}

  seed_list = [seed, 1001, 0, 0]
  if distributed:
    import horovod.tensorflow as hvd
    seed2 = hvd.rank() * 1234 + np.array(seed_list)
  else:
    seed2 = seed_list
  for k, s in zip(split_list, seed2):
    sampler_dict[k] = get_sampler(data_config.base_sampler, s)

  # Wrap it with hierarchical sampler.
  if data_config.hierarchical:
    for k, s in zip(split_list, seed2):
      if data_config.blender in ['hard']:
        blender = get_blender(data_config.blender)
      elif data_config.blender in ['blur']:
        blender = get_blender(
            data_config.blender,
            window_size=data_config.blur_window_size,
            stride=data_config.blur_stride,
            nrun=data_config.blur_nrun,
            seed=s)
      elif data_config.blender in ['markov-switch']:
        blender = get_blender(
            data_config.blender,
            base_dist=np.ones([data_config.nstage]) / float(
                data_config.nstage),
            switch_prob=data_config.markov_switch_prob,
            seed=s)
      else:
        raise ValueError('Unknown blender {}'.format(data_config.blender))

      # Mix class hierarchy and non class hierarchy.
      if data_config.mix_class_hierarchy:
        sampler_dict[k] = MixSampler([
            HierarchicalEpisodeSampler(sampler_dict[k], blender, False,
                                       data_config.use_new_class_hierarchy,
                                       data_config.use_same_family,
                                       data_config.shuffle_time, s),
            HierarchicalEpisodeSampler(sampler_dict[k], blender, True,
                                       data_config.use_new_class_hierarchy,
                                       data_config.use_same_family,
                                       data_config.shuffle_time, s + 1)
        ], [0.5, 0.5], 1023)  # Set for 0.5/0.5 for now.
      else:
        sampler_dict[k] = HierarchicalEpisodeSampler(
            sampler_dict[k], blender, data_config.use_class_hierarchy,
            data_config.use_new_class_hierarchy, data_config.use_same_family,
            data_config.shuffle_time, s)
      kwargs['nstage'] = data_config.nstage

  # Wrap it with continual sampler.
  if data_config.continual:
    assert not data_config.hierarchical
    split_list = ['train_fs', 'trainval_fs']
    for k, s in zip(split_list, seed2):
      sampler_dict[k] = ContinualEpisodeSampler(sampler_dict[k], s)
      assert stage is not None
      sampler_dict[k].set_stage(stage)

  # Wrap it with semisupervised sampler.
  if data_config.semisupervised:
    for k, s in zip(split_list, seed2):
      sampler_dict[k] = SemiSupervisedEpisodeSampler(sampler_dict[k], s)
      kwargs['label_ratio'] = data_config.label_ratio
      kwargs['nd'] = data_config.distractor_nway
      kwargs['nshotd'] = data_config.distractor_nshot
      kwargs['md'] = data_config.distractor_nquery
  elif data_config.unsupervised:
    for k, s in zip(split_list, seed2):
      sampler_dict[k] = UnsupervisedEpisodeSampler(sampler_dict[k], s)
      kwargs['label_ratio'] = data_config.label_ratio

  # Random background.
  if data_config.random_background != 'none':
    if data_config.random_background in ['uppsala', 'uppsala_double']:
      folder = './data/uppsala-texture'
      bg_dataset = [
          UppsalaDataset(folder, s) for s in ['train', 'train', 'val', 'test']
      ]
    else:
      assert False

    if data_config.random_background == 'uppsala':
      Processor = BackgroundProcessor
    elif data_config.random_background == 'uppsala_double':
      Processor = BackgroundProcessorV2
    bg_random = [True, False, False, False]
    bg_processor_dict = dict(
        zip(split_list, [
            Processor(
                d,
                random=r,
                random_apply=data_config.random_background_random_apply,
                apply_prob=data_config.random_background_apply_prob,
                random_context=data_config.random_background_random_context,
                gaussian_noise_std=data_config.random_background_gaussian_std)
            for d, r in zip(bg_dataset, bg_random)
        ]))
  else:
    bg_processor_dict = dict(zip(split_list, [None] * len(split_list)))

  norm_prep = FloatPreprocessor()
  da_prep = DataAugmentationPreprocessor(md.image_size, md.crop_size,
                                         md.random_crop, md.random_flip,
                                         md.random_color, md.random_rotate)
  nchw_prep = NCHWPreprocessor()
  random_box_prep = RandomBoxOccluder()

  train_prep_list = [da_prep]
  val_prep_list = []
  if random_box:
    train_prep_list.append(random_box_prep)
    val_prep_list.append(random_box_prep)
  train_prep_list.append(norm_prep)
  val_prep_list.append(norm_prep)
  if nchw:
    train_prep_list.append(nchw_prep)
    val_prep_list.append(nchw_prep)
  train_prep = SequentialPreprocessor(*train_prep_list)
  val_prep = SequentialPreprocessor(*val_prep_list)

  prep = {
      'train_fs': train_prep,
      'trainval_fs': val_prep,
      'val_fs': val_prep,
      'test_fs': val_prep
  }

  # Adds data iterators.
  if data_config.semisupervised:
    IteratorClassList = [SemiSupervisedEpisodeIterator] * 4
  elif (data_config.unsupervised and data_config.augmentation and
        data_config.imbalance):
    IteratorClassList = [UnsupervisedAugmentImbalanceEpisodeIterator]
    IteratorClassList = IteratorClassList + [UnsupervisedEpisodeIterator] * 3
  elif data_config.unsupervised and data_config.augmentation:
    IteratorClassList = [UnsupervisedAugmentEpisodeIterator]
    IteratorClassList = IteratorClassList + [UnsupervisedEpisodeIterator] * 3
  elif data_config.unsupervised:
    IteratorClassList = [UnsupervisedEpisodeIterator] * 4
  else:
    IteratorClassList = [EpisodeIterator] * 4

  it_dict = {}

  # Key, batch size
  for k, b, clas in zip(split_list, [batch_size, 1, 1, 1], IteratorClassList):
    if data[k] is not None:
      it_dict[k] = clas(
          data[k],
          data_config,
          sampler_dict[k],
          batch_size=b,
          preprocessor=prep[k],
          episode_processor=bg_processor_dict[k],
          prefetch=prefetch,
          save_additional_info=save_additional_info,
          **kwargs)
    else:
      it_dict[k] = None
  return it_dict


def get_dataiter_vid(data,
                     data_config,
                     batch_size=1,
                     prefetch=True,
                     distributed=False,
                     nchw=True,
                     seed=0):
  """Gets video episode iterator (e.g. for SAY-Cam)."""
  # Split.
  split_list = ['train_fs']
  seed_list = [seed]
  if distributed:
    import horovod.tensorflow as hvd
    seed2 = hvd.rank() * 1234 + np.array(seed_list)
  else:
    seed2 = seed_list

  norm_prep = FloatPreprocessor()

  # Adds data iterators.
  if data_config.augmentation:
    IteratorClassList = [UnsupervisedAugmentVidEpisodeIterator]
  else:
    assert False, "Not supported"

  it_dict = {}

  random_list = [True]
  # Key, batch size
  for k, b, r, s, clas in zip(split_list, [batch_size], random_list, seed2,
                              IteratorClassList):
    # Changed to No flip due to online sequential nature.
    if data[k] is not None:
      it_dict[k] = clas(
          data[k],
          data_config,
          batch_size=b,
          preprocessor=norm_prep,
          prefetch=prefetch,
          data_format="NCHW" if nchw else "NHWC",
          seed=s)
    else:
      it_dict[k] = None
  return it_dict


def get_dataiter_sim(data,
                     data_config,
                     batch_size=1,
                     nchw=True,
                     prefetch=True,
                     distributed=False,
                     same_seed=False,
                     seed=0,
                     iid=False):
  """Gets few-shot episode iterator in simulated environment.

  Args:
    data: Object. Dataset object.
    data_config: Config. Data source episode config.
    batch_size: Int. Batch size.
    nchw: Bool. Whether to transpose the images to NCHW.
    prefetch: Bool. Whether to add prefetching module in the data loader.
  """
  md = data['metadata']
  data['trainval_fs'] = data['train_fs']  # Use the same for trainval.

  # Split.
  split_list = ['train_fs', 'trainval_fs', 'val_fs', 'test_fs']

  # Gets episodic sampler.
  sampler_dict = {}

  seed_list = [seed, 1001, 0, 0]
  if distributed:
    if same_seed:
      seed2 = np.array(seed_list)
    else:
      import horovod.tensorflow as hvd
      seed2 = hvd.rank() * 1234 + np.array(seed_list)
  else:
    seed2 = seed_list

  cycle_list = [True, False, False, False]
  shuffle_list = [True, False, False, False]

  for k, s, cyc, shuf in zip(split_list, seed2, cycle_list, shuffle_list):
    sampler_dict[k] = MinibatchSampler(s, cycle=cyc, shuffle=shuf)
  norm_prep = FloatPreprocessor()

  # Adds data iterators.
  if data_config.augmentation:
    if iid:
      IteratorClassList = [MixUnsupAugmentSimEpisodeIterator]
      # assert False
    else:
      if same_seed:
        IteratorClassList = [UnsupervisedAugmentSimEpisodeIteratorMultiGPU]
      else:
        IteratorClassList = [UnsupervisedAugmentSimEpisodeIterator]
    # TODO: here labels are not removed during evaluation.
    IteratorClassList = IteratorClassList + [SimEpisodeIterator] * 3
  else:
    IteratorClassList = [SimEpisodeIterator] * 4

  it_dict = {}

  random_list = [True, False, False, False]
  # Key, batch size
  for k, b, r, s, clas in zip(['train_fs', 'trainval_fs', 'val_fs', 'test_fs'],
                              [batch_size, 1, 1, 1], random_list, seed2,
                              IteratorClassList):
    if distributed:
      import horovod.tensorflow as hvd
      kwargs = {'rank': hvd.rank(), 'totalrank': hvd.size()}
    else:
      kwargs = {}
    # Changed to No flip due to online sequential nature.
    if data[k] is not None:
      it_dict[k] = clas(
          data[k],
          data_config,
          sampler_dict[k],
          batch_size=b,
          preprocessor=norm_prep,
          prefetch=prefetch,
          random_crop=r,
          random_drop=r,
          random_flip=False,
          random_jitter=False,
          random_shuffle_objects=r,
          data_format="NCHW" if nchw else "NHWC",
          seed=s,
          **kwargs)
    else:
      it_dict[k] = None
  return it_dict


def get_dataiter_matterport_category(data,
                                     batch_size,
                                     nchw=False,
                                     data_aug=False,
                                     distributed=False,
                                     seed=0):
  md = data['metadata']
  key_list = ['train', 'val', 'test']
  norm_prep = FloatPreprocessor()
  da_prep = DataAugmentationPreprocessor(md.image_size, md.crop_size,
                                         md.random_crop, md.random_flip,
                                         md.random_color, md.random_rotate)
  if nchw:
    nchw_prep = NCHWPreprocessor()
    if data_aug:
      train_prep = SequentialPreprocessor(da_prep, norm_prep, nchw_prep)
    else:
      train_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    val_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    test_prep = SequentialPreprocessor(norm_prep, nchw_prep)
  else:
    if data_aug:
      train_prep = norm_prep
    else:
      train_prep = norm_prep
    val_prep = norm_prep
    test_prep = norm_prep

  prep = {'train': train_prep, 'val': val_prep, 'test': test_prep}
  it_dict = {}
  for k in key_list:
    if distributed:
      import horovod.tensorflow as hvd
      kwargs = {'rank': hvd.rank(), 'totalrank': hvd.size()}
    else:
      kwargs = {}

    config = EpisodeConfig()
    config.maxlen = 200
    it_dict[k] = MatterportCategoryIterator(
        data[k],
        config,
        batch_size,
        prefetch=True,
        preprocessor=prep[k],
        seed=seed,
        **kwargs)
  return it_dict


def get_dataiter_epoch(data,
                       batch_size,
                       nchw=False,
                       shuffle=True,
                       frame_rate=1,
                       distributed=False,
                       data_aug=False,
                       use_file=False,
                       seed=0):
  """Gets dataset iterator for epoch-based supervised learning."""
  md = data['metadata']
  key_list = ['train', 'val', 'test']
  norm_prep = FloatPreprocessor()
  da_prep = DataAugmentationPreprocessor(md.image_size, md.crop_size,
                                         md.random_crop, md.random_flip,
                                         md.random_color, md.random_rotate)
  if nchw:
    nchw_prep = NCHWPreprocessor()
    if data_aug:
      train_prep = SequentialPreprocessor(da_prep, norm_prep, nchw_prep)
    else:
      train_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    val_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    test_prep = SequentialPreprocessor(norm_prep, nchw_prep)
  else:
    if data_aug:
      train_prep = norm_prep
    else:
      train_prep = norm_prep
    val_prep = norm_prep
    test_prep = norm_prep

  prep = {'train': train_prep, 'val': val_prep, 'test': test_prep}
  drop_remainder = {'train': True, 'val': False, 'test': False}
  it_dict = {}

  for k in key_list:
    if distributed:
      import horovod.tensorflow as hvd
      kwargs = {'rank': hvd.rank(), 'totalrank': hvd.size()}
    else:
      kwargs = {}
    if use_file:
      cls = EpochFileMinibatchIterator
    else:
      cls = EpochMinibatchIteratorV2
    it_dict[k] = cls(
        data[k],
        batch_size,
        prefetch=True,
        shuffle=shuffle,
        preprocessor=prep[k],
        drop_remainder=drop_remainder[k],
        seed=seed,
        frame_rate=frame_rate,
        **kwargs)
  return it_dict


def get_dataiter_contrastive(data,
                             batch_size,
                             nchw=False,
                             distributed=False,
                             color_distort=True,
                             flip=True,
                             color_distort_strength=1.0,
                             area_range_lb=0.08,
                             min_object_covered=0.1,
                             num_views=2,
                             aug=False,
                             cycle=False,
                             sim=False,
                             vid=False,
                             background_ratio=0.0,
                             only_one=False,
                             non_iid=False,
                             non_iid_data_config=None,
                             seed=0):
  """Gets dataset iterator for contrastive learning."""
  md = data['metadata']
  key_list = ['train']
  norm_prep = FloatPreprocessor()

  if nchw:
    nchw_prep = NCHWPreprocessor()
    train_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    val_prep = SequentialPreprocessor(norm_prep, nchw_prep)
    test_prep = SequentialPreprocessor(norm_prep, nchw_prep)
  else:
    train_prep = norm_prep
    val_prep = norm_prep
    test_prep = norm_prep

  prep = {'train': train_prep, 'val': val_prep, 'test': test_prep}
  drop_remainder = {'train': True, 'val': False, 'test': False}
  it_dict = {}

  for k in key_list:
    # assert distributed
    if distributed:
      import horovod.tensorflow as hvd
      kwargs = {'rank': hvd.rank(), 'totalrank': hvd.size()}
    else:
      kwargs = {}

    if sim:
      if non_iid:
        it_dict[k] = MixUnsupAugmentSimIteratorNonIID(
            data[k],
            non_iid_data_config,
            batch_size,
            preprocessor=prep[k],
            seed=seed,
            **kwargs)
      else:
        it_dict[k] = MixUnsupAugmentSimIterator(
            data[k],
            non_iid_data_config,
            batch_size,
            prefetch=True,
            preprocessor=prep[k],
            seed=seed,
            **kwargs)
    elif vid:
      # Video dataset
      if non_iid:
        it_dict[k] = MixUnsupAugmentVidIteratorNonIID(
            data[k],
            non_iid_data_config,
            batch_size,
            preprocessor=prep[k],
            seed=seed,
            **kwargs)
      else:
        it_dict[k] = MixUnsupAugmentVidIterator(
            data[k],
            non_iid_data_config,
            batch_size,
            preprocessor=prep[k],
            seed=seed,
            **kwargs)
    else:
      if not aug:
        if background_ratio > 0.0:
          it_dict[k] = SIMCLRImbalanceMinibatchIterator(
              data[k],
              batch_size,
              prefetch=True,
              preprocessor=prep[k],
              drop_remainder=drop_remainder[k],
              color_distort=color_distort,
              flip=flip,
              color_distort_strength=color_distort_strength,
              area_range_lb=area_range_lb,
              min_object_covered=min_object_covered,
              seed=seed,
              num_views=num_views,
              cycle=cycle,
              background_ratio=background_ratio,
              only_one=only_one,
              **kwargs)
        else:
          if non_iid:
            data_config = non_iid_data_config
            sampler_kwargs = {
                'alpha': data_config.crp_alpha,
                'theta': data_config.crp_theta,
                'allow_repeat': False,
                'max_num': data_config.maxlen,
                'max_num_per_cls': data_config.max_num_per_cls,
                # 'nstage': data_config.nstage
            }
            data_config.num_aug = num_views
            sampler = get_sampler(data_config.base_sampler, seed=0)

            # Wrap it with hierarchical sampler.
            if data_config.hierarchical:
              blender = get_blender(
                  data_config.blender,
                  base_dist=np.ones([data_config.nstage]) / float(
                      data_config.nstage),
                  switch_prob=data_config.markov_switch_prob,
                  seed=0)
              # Mix class hierarchy and non class hierarchy.
              sampler_kwargs['nstage'] = data_config.nstage
              sampler = HierarchicalEpisodeSampler(
                  sampler,
                  blender,
                  data_config.use_class_hierarchy,
                  data_config.use_new_class_hierarchy,
                  data_config.use_same_family,
                  data_config.shuffle_time,
                  seed=0)
            it_dict[k] = UnsupervisedAugmentEpisodeIteratorNonIID(
                data[k],
                data_config,
                sampler,
                batch_size,
                preprocessor=None,
                episode_processor=None,
                prefetch=True,
                **sampler_kwargs)
            pass
          else:
            it_dict[k] = SIMCLRMinibatchIterator(
                data[k],
                batch_size,
                prefetch=True,
                preprocessor=prep[k],
                drop_remainder=drop_remainder[k],
                color_distort=color_distort,
                flip=flip,
                color_distort_strength=color_distort_strength,
                area_range_lb=area_range_lb,
                min_object_covered=min_object_covered,
                seed=seed,
                num_views=num_views,
                cycle=cycle,
                **kwargs)
      else:
        it_dict[k] = SIMCLRAugMinibatchIterator(
            data[k],
            batch_size,
            prefetch=True,
            preprocessor=prep[k],
            drop_remainder=drop_remainder[k],
            color_distort=color_distort,
            flip=flip,
            color_distort_strength=color_distort_strength,
            area_range_lb=area_range_lb,
            min_object_covered=min_object_covered,
            seed=seed,
            num_views=num_views,
            cycle=cycle,
            **kwargs)
  return it_dict
