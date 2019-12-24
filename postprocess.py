import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

import params

class Postprocess(Layer):
    def __init__(self, output_shape=None, **kw):
        self.emb_shape = output_shape
        super().__init__(**kw)

    def build(self, input_shape):
        input_shape = tuple(int(x) for x in tuple(input_shape)[1:])
        emb_shape = (self.emb_shape,) if self.emb_shape else input_shape

        self.pca_matrix = self.add_weight(name='pca_matrix', shape=emb_shape + input_shape)
        self.pca_means = self.add_weight( name='pca_means', shape=input_shape + (1,))

    def call(self, x):

        x = K.dot(self.pca_matrix, (K.transpose(x) - self.pca_means))
        x = K.transpose(x)

        x = tf.clip_by_value(x, params.QUANTIZE_MIN_VAL, params.QUANTIZE_MAX_VAL)
        x = ((x - params.QUANTIZE_MIN_VAL) *
             (255.0 / (params.QUANTIZE_MAX_VAL - params.QUANTIZE_MIN_VAL)))
        return K.cast(x, 'uint8')