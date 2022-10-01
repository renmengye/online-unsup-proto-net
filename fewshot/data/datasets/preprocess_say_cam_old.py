# Original author provided the license below. The code is modified by Mengye
# Ren.
#
# MIT License

# Copyright (c) 2021 Emin Orhan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Read SAYCam videos')
parser.add_argument('data', metavar='DIR', help='path to SAYCam videos')
parser.add_argument('--save-dir', default='', type=str, help='save directory')
parser.add_argument(
    '--fps', default=5, type=int, help='sampling rate (frames per second)')
parser.add_argument(
    '--seg-len', default=288, type=int, help='segment length (seconds)')

if __name__ == '__main__':

  args = parser.parse_args()

  file_list = os.listdir(args.data)
  file_list.sort()
  print(file_list)

  class_counter = 0
  img_counter = 0
  file_counter = 0

  final_size = 224
  resized_minor_length = 256
  edge_filter = False
  n_imgs_per_class = args.seg_len * args.fps

  curr_dir_name = os.path.join(args.save_dir,
                               'class_{:04d}'.format(class_counter))
  os.mkdir(curr_dir_name)

  for file_indx in file_list:
    file_name = os.path.join(args.data, file_indx)

    cap = cv2.VideoCapture(file_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # take every sample_rate frames (30: 1fps, 15: 2fps, 10: 3fps, 6: 5fps,
    # 5: 6fps, 3: 10fps, 2: 15fps, 1: 30fps)
    sample_rate = frame_rate // args.fps + 1

    print('Total frame count: ', frame_count)
    print('Native frame rate: ', frame_rate)

    fc = 0
    ret = True

    # Resize
    new_height = frame_height * resized_minor_length // min(
        frame_height, frame_width)
    new_width = frame_width * resized_minor_length // min(
        frame_height, frame_width)

    while (fc < frame_count):

      ret, frame = cap.read()

      if fc % sample_rate == 0 and ret:

        # Resize
        resized_frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Crop
        height, width, _ = resized_frame.shape
        startx = width // 2 - (final_size // 2)
        starty = height // 2 - (final_size // 2) - 16
        cropped_frame = resized_frame[starty:starty +
                                      final_size, startx:startx + final_size]
        assert cropped_frame.shape[0] == final_size and \
            cropped_frame.shape[1] == final_size, \
            (cropped_frame.shape, height, width)

        if edge_filter:
          cropped_frame = cv2.Laplacian(cropped_frame, cv2.CV_64F, ksize=5)
          img_min = cropped_frame.min()
          img_max = cropped_frame.max()
          cropped_frame = np.uint8(
              255 * (cropped_frame - img_min) / (img_max - img_min))

        cv2.imwrite(
            os.path.join(curr_dir_name, 'img_{:04d}.jpeg'.format(img_counter)),
            cropped_frame[::-1, ::-1, :])
        img_counter += 1

        if img_counter == n_imgs_per_class:
          img_counter = 0
          class_counter += 1
          curr_dir_name = os.path.join(args.save_dir,
                                       'class_{:04d}'.format(class_counter))
          os.mkdir(curr_dir_name)

      fc += 1

    cap.release()

    file_counter += 1
    print('Completed video {:4d} of {:4d}'.format(file_counter,
                                                  len(file_list)))
