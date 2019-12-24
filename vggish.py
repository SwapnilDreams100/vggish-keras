import os
import logging
from functools import partial

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K

from postprocess import Postprocess
import params

log = logging.getLogger(__name__)


def VGGish(pump=None,
           input_shape=None,
           include_top=False,
           pooling='avg',
           weights='audioset',
           name='vggish',
           compress=False):

    with tf.name_scope(name):
        if input_shape:
            pass

        elif pump:
            inputs = pump.layers('tf.keras')[params.PUMP_INPUT]

        elif include_top:
            input_shape = params.NUM_FRAMES, params.NUM_BANDS, 1

        else:
            input_shape = None, None, 1

        # use input_shape to make input
        if input_shape:
            inputs = kl.Input(shape=input_shape, name='input_1')

        # setup layer params
        conv = partial(
            kl.Conv2D,
            kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')

        maxpool = partial(
            kl.MaxPooling2D, pool_size=(2, 2), strides=(2, 2), padding='same')

        # Block 1
        x = conv(64, name='conv1')(inputs)
        x = maxpool(name='pool1')(x)

        # Block 2
        x = conv(128, name='conv2')(x)
        x = maxpool(name='pool2')(x)

        # Block 3
        x = conv(256, name='conv3/conv3_1')(x)
        x = conv(256, name='conv3/conv3_2')(x)
        x = maxpool(name='pool3')(x)

        # Block 4
        x = conv(512, name='conv4/conv4_1')(x)
        x = conv(512, name='conv4/conv4_2')(x)
        x = maxpool(name='pool4')(x)

        if include_top:
            dense = partial(kl.Dense, activation='relu')

            # FC block
            x = kl.Flatten(name='flatten_')(x)
            x = dense(4096, name='fc1/fc1_1')(x)
            x = dense(4096, name='fc1/fc1_2')(x)
            x = dense(params.EMBEDDING_SIZE, name='fc2')(x)

            if compress:
                x = Postprocess()(x)
        else:
            globalpool = (
                kl.GlobalAveragePooling2D() if pooling == 'avg' else
                kl.GlobalMaxPooling2D() if pooling == 'max' else None)

            if globalpool:
                x = globalpool(x)

        # Create model
        model = Model(inputs, x, name='model')
        load_vggish_weights(model, weights, strict=bool(weights))
    return model

def load_vggish_weights(model, weights, strict=False):
    # lookup weights location
    if weights in params.WEIGHTS_PATHS:
        w_name, weights = weights, params.WEIGHTS_PATHS[weights]

    # load weights
    if weights:
        model.load_weights(weights, by_name=True)
    return model