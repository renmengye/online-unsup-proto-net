"""A sampler for semi-supervised collections.

Author: Mengye Ren (mren@cs.toronto.edu)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from fewshot.data.registry import RegisterSampler
from fewshot.data.samplers.semisupervised_episode_sampler import SemiSupervisedEpisodeSampler  # NOQA
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterSampler('unsupervised')
class UnsupervisedEpisodeSampler(SemiSupervisedEpisodeSampler):

  def set_stage(self, stage):
    return self.sampler.set_stage(stage)

  def sample_label_mask(self, cls_support, label_ratio, const=0.5):
    """See SemiSupervisedEpisodeSampler for documentation."""
    flag = np.zeros([len(cls_support)])
    return flag
