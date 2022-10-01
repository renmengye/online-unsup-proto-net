import os
import h5py
import numpy as np
from tqdm import tqdm


def _make_iter(img_arr, l_arr, start=0, end=None, interval=1):
  prev = 0
  if end is None:
    end = len(l_arr)
  l_cum = np.cumsum(l_arr)
  for i, idx in enumerate(l_cum[start:end:interval]):
    if i > 0:
      prev = l_cum[i - 1]
    yield img_arr[prev:idx]


def main():
  ratio = 50
  # input_folder = '/mnt/research/datasets/say-cam/h5_data/{}'
  # output_folder = '/mnt/research/datasets/say-cam/h5_data_{}s/{}'.format(
  #     288 * ratio, '{}')
  input_folder = '/scratch/hdd001/home/mren/say-cam/h5_data/{}'
  output_folder = '/scratch/hdd001/home/mren/say-cam/h5_data_{}s/{}'.format(
      288 * ratio, '{}')
  for split in ['S']:
    input_folder_ = input_folder.format(split)
    output_folder_ = output_folder.format(split)
    if not os.path.exists(output_folder_):
      os.makedirs(output_folder_)

    counter = 1
    file_counter = 0
    images_all = []
    for f in tqdm(os.listdir(input_folder_), desc=split, ncols=0):
      with h5py.File(os.path.join(input_folder_, f), 'r') as h5f:
        images = h5f['images'][:]
        images_len = h5f['images_len'][:]
        # print(images.shape, images_len.shape)
        images_iter = _make_iter(images, images_len)
        max_len = max(images_len)
        images_all.extend(images_iter)
        file_counter += 1

      if file_counter == ratio:
        f2 = "session_{:06d}.h5".format(counter)
        print(f2)
        with h5py.File(os.path.join(output_folder_, f2), 'w') as h5f2:
          # print(len(images_all))
          dset = h5f2.create_dataset(
              "images",
              shape=(len(images_all),),
              dtype=h5py.vlen_dtype(np.dtype(np.uint8)))
          for i, im in enumerate(images_all):
            dset[i] = list(im[:, 0])
        counter += 1
        file_counter = 0
        images_all = []


if __name__ == '__main__':
  main()
  # fix_label()
