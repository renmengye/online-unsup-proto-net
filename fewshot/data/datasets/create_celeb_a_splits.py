from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import numpy as np
import os
import pandas as pd
import pickle as pkl
from tqdm import tqdm


def read_attr(path):
  return pd.read_csv(path, sep="\s+", header=1)


def split(df):
  ntrain = 6000
  nval = 2000
  ntest = 2177
  df = np.array(df)
  identity = df[:, 1].astype(np.int64) - 1

  rnd = np.random.RandomState(0)
  split = np.arange(identity.max())
  rnd.shuffle(split)
  train_id = np.sort(split[:ntrain])
  val_id = np.sort(split[ntrain:ntrain + nval])
  test_id = np.sort(split[ntrain + nval:ntrain + nval + ntest])

  train_id_set = set(train_id)
  val_id_set = set(val_id)
  test_id_set = set(test_id)

  train_idx_list = []
  val_idx_list = []
  test_idx_list = []

  for i in range(df.shape[0]):
    if df[i, 1] in train_id_set:
      train_idx_list.append(i)
    elif df[i, 1] in val_id_set:
      val_idx_list.append(i)
    elif df[i, 1] in test_id_set:
      test_idx_list.append(i)

  train_idx_list = np.array(train_idx_list)
  val_idx_list = np.array(val_idx_list)
  test_idx_list = np.array(test_idx_list)

  rnd.shuffle(train_idx_list)

  traintrain = int(np.floor(0.8 * len(train_idx_list)))
  trainval = int(np.floor(0.1 * len(train_idx_list)))
  traintrain_idx_list = train_idx_list[:traintrain]
  trainval_idx_list = train_idx_list[traintrain:traintrain + trainval]
  traintest_idx_list = train_idx_list[traintrain + trainval:]
  traintrain_idx_list = np.sort(traintrain_idx_list)
  trainval_idx_list = np.sort(trainval_idx_list)
  traintest_idx_list = np.sort(traintest_idx_list)

  return {
      "train": traintrain_idx_list,
      "val": val_idx_list,
      "test": test_idx_list,
      "trainval": trainval_idx_list,
      "traintest": traintest_idx_list
  }


def main():
  df = read_attr(os.path.join(args.data_folder, "identity_CelebA.txt"))
  split_info = split(df)
  for sp in ["val", "test", "train", "trainval", "traintest"]:
    with open(os.path.join(args.data_folder, "{}.txt".format(sp)), 'w') as f:
      for i in split_info[sp]:
        f.write(str(i) + '\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Build index")
  parser.add_argument('--data_folder', type=str, default=None)
  args = parser.parse_args()
  main()
