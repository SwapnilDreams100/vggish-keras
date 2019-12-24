# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Post-process embeddings from VGGish."""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

from . import params

class Postprocess(Layer):
    def __init__(self, output_shape=None, **kw):
        self.emb_shape = output_shape
        super().__init__(**kw)

    def build(self, input_shape):
        input_shape = tuple(int(x) for x in tuple(input_shape)[1:])
        emb_shape = (self.emb_shape,) if self.emb_shape else input_shape

        self.pca_matrix = self.add_weight(
            name='pca_matrix', shape=emb_shape + input_shape)
        self.pca_means = self.add_weight(
            name='pca_means', shape=input_shape + (1,))

    def call(self, x):
        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        x = K.dot(self.pca_matrix, (K.transpose(x) - self.pca_means))
        x = K.transpose(x)

        # Quantize by:
        # - clipping to [min, max] range
        # - convert to 8-bit in range [0.0, 255.0]
        # - cast 8-bit float to uint8
        x = tf.clip_by_value(x, params.QUANTIZE_MIN_VAL, params.QUANTIZE_MAX_VAL)
        x = ((x - params.QUANTIZE_MIN_VAL) *
             (255.0 / (params.QUANTIZE_MAX_VAL - params.QUANTIZE_MIN_VAL)))
        return K.cast(x, 'uint8')
