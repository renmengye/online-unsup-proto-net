import tensorflow as tf
from fewshot.data.preprocessors.preprocessor import Preprocessor


class FloatPreprocessor(Preprocessor):
  """Normalization preprocessor, subtract mean and divide variance."""

  def preprocess(self, inputs):
    return tf.image.convert_image_dtype(inputs, tf.float32)
