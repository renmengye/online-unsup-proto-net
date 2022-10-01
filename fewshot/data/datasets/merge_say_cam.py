import os
import h5py
import numpy as np
from tqdm import tqdm


def main():
  ratio = 50
  input_folder = '/mnt/research/datasets/say-cam/h5_data/{}'
  output_folder = '/mnt/research/datasets/say-cam/h5_data_{}s/{}'.format(
      288 * ratio, '{}')
  for split in ['S']:
    input_folder_ = input_folder.format(split)
    output_folder_ = output_folder.format(split)
    if not os.path.exists(output_folder_):
      os.makedirs(output_folder_)

    counter = 1
    images_all = []
    images_len_all = []
    for f in tqdm(os.listdir(input_folder_), desc=split):
      with h5py.File(os.path.join(input_folder_, f), 'r') as h5f:
        images = h5f['images'][:]
        images_len = h5f['images_len'][:]
        # print(images.shape, images_len.shape)
        images_all.append(images)
        images_len_all.append(images_len)

      if len(images_all) == ratio:
        f2 = "session_{:06d}.h5".format(counter)
        print(f2)
        with h5py.File(os.path.join(output_folder_, f2), 'w') as h5f2:
          h5f2['images'] = np.concatenate(images_all)
          h5f2['images_len'] = np.concatenate(images_len_all)
        counter += 1
        images_all = []
        images_len_all = []


if __name__ == '__main__':
  main()
  # fix_label()
