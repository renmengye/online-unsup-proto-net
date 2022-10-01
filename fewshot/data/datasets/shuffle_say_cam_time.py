import os
import h5py
import numpy as np
from tqdm import tqdm


def _make_iter(img_arr, l_arr):
  """Makes an PNG encoding string iterator."""
  prev = 0
  l_cum = np.cumsum(l_arr)
  for i, idx in enumerate(l_cum):
    yield img_arr[prev:idx]
    prev = idx


def fix_label():
  import glob
  files = glob.glob('/mnt/research/datasets/say-cam/h5_data_shuffle/*/*.h5')
  for f in tqdm(files):
    with h5py.File(f, 'a') as h5f:
      labels = h5f['labels'][:] - 1
      del h5f['labels']
      h5f['labels'] = labels


def main():
  input_folder = '/mnt/research/datasets/say-cam/h5_data/{}'
  output_folder = '/mnt/research/datasets/say-cam/h5_data_shuffle/{}'
  for split in ['S', 'A', 'Y']:
    input_folder_ = input_folder.format(split)
    output_folder_ = output_folder.format(split)
    if not os.path.exists(output_folder_):
      os.makedirs(output_folder_)
    all_data = []
    for f in tqdm(os.listdir(input_folder_), desc=split + ' read'):
      label = int(f.split('_')[1].split('.')[0]) - 1  # 0-based.
      with h5py.File(os.path.join(input_folder_, f), 'r') as h5f:
        images = h5f['images'][:]
        images_len = h5f['images_len'][:]
      for im in _make_iter(images, images_len):
        all_data.append((im, label))
    rnd = np.random.RandomState(0)
    rnd.shuffle(all_data)
    num_per_shard = 1440
    num_shard = int(np.ceil(len(all_data) / num_per_shard))
    for s in tqdm(range(num_shard), desc=split + ' write'):
      start = s * num_per_shard
      end = (s + 1) * num_per_shard
      data = all_data[start:end]
      d_images = [d[0] for d in data]
      d_labels = [d[1] for d in data]
      with h5py.File(
          os.path.join(output_folder_, 'shard_{:06d}.h5'.format(s)),
          'w') as h5f:
        h5f['images'] = np.concatenate(d_images)
        h5f['images_len'] = np.array([len(d) for d in d_images],
                                     dtype=np.int64)
        h5f['labels'] = np.array(d_labels)


if __name__ == '__main__':
  main()
  # fix_label()
