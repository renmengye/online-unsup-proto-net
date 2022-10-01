from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class dummy_context_mgr():

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False
