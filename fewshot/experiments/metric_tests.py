""""Unit tests for blender."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest
import numpy as np

from fewshot.experiments.metrics import convert_pred_to_full
from fewshot.experiments.metrics import calc_rand
from fewshot.experiments.metrics import calc_homogeneity


class MetricTests(unittest.TestCase):

  def test_convert(self):
    y_pred = np.array([[
        50, 50, 50, 50, 3, 3, 50, 50, 3, 50, 3, 6, 7, 7, 7, 50, 7, 6, 7, 9, 6,
        8, 50, 50, 6, 50, 10, 10, 50, 50, 15, 15, 50, 50, 50, 11, 13, 50, 1,
        20, 50, 20, 1, 50, 15, 15, 15, 16, 50, 23, 23, 18, 24, 23, 17, 11, 24,
        20, 16, 16, 50, 16, 23, 6, 50, 50, 20, 28, 22, 21, 0, 28, 0, 22, 11,
        50, 0, 29, 50, 50, 6, 6, 6, 29, 33, 6, 16, 50, 50, 27, 38, 27, 33, 33,
        50, 0, 30, 32, 32, 24, 23, 23, 29, 29, 34, 50, 50, 27, 43, 27, 50, 29,
        21, 50, 34, 34, 46, 34, 34, 50, 24, 50, 24, 50, 50, 49, 4, 49, 4, 48,
        50, 5, 4, 31, 22, 32, 50, 50, 49, 48, 48, 48, 27, 43, 43, 28, 50, 43,
        32, 9
    ]])
    y_f = convert_pred_to_full(y_pred)
    print('y_f', y_f)

    y_full = np.array([[
        0, 1, 2, 2, 2, 2, 3, 4, 2, 5, 2, 4, 5, 5, 5, 3, 5, 4, 5, 4, 4, 3, 6, 7,
        4, 8, 3, 3, 9, 10, 10, 10, 11, 12, 13, 6, 8, 14, 1, 1, 15, 1, 1, 16,
        10, 10, 10, 11, 17, 17, 17, 18, 18, 17, 12, 19, 18, 20, 11, 11, 21, 11,
        22, 19, 23, 24, 1, 25, 16, 15, 0, 25, 0, 16, 0, 26, 0, 23, 12, 27, 19,
        19, 19, 23, 12, 19, 11, 28, 29, 20, 29, 20, 12, 12, 30, 0, 24, 26, 26,
        18, 17, 17, 23, 23, 27, 21, 31, 20, 21, 20, 32, 23, 33, 13, 27, 27, 13,
        27, 27, 34, 18, 35, 18, 36, 37, 35, 37, 35, 37, 34, 38, 38, 37, 15, 16,
        26, 39, 39, 35, 34, 34, 34, 20, 21, 21, 21, 31, 26, 26, 40
    ]])
    r = calc_rand([{'pred_id': y_pred, 'y_full': y_full}])
    print('rand', r)
    r = calc_homogeneity([{'pred_id': y_pred, 'y_full': y_full}])
    print('homogeneity', r)


if __name__ == '__main__':
  unittest.main()
