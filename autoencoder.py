from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from functools import reduce
from icecream import ic
from collections import namedtuple
from typing import Tuple, Optional, Union, Any


class EncoderCNN(tf.keras.Model, ABC):
    def __init__(self, input_size: Tuple[int, int, int], code_length: int):
        super(EncoderCNN, self).__init__()
        self.code_length = code_length
        self.encoder_layers = []
        # Input layer
        self.input_size = input_size
        # self.encoder_layers.append(layers.InputLayer(input_shape=input_size))
        self.encoder_layers.append(layers.BatchNormalization())
        # Convolution layers
        self.input_min_size = min(input_size[:-1])
        convolution_layer_count = int(np.log2(self.input_min_size / code_length)) + 1
        conv_params = namedtuple('conv_params', ['kernel_size', 'filters'])
        self.conv_params = [conv_params(kernel_size=3, filters=convolution_layer_count ** 2 // 3)
                            for layer_ind in range(convolution_layer_count - 1)]
        for param in self.conv_params:
            self.encoder_layers.append(
                layers.Conv2D(kernel_size=param.kernel_size, filters=param.filters,
                              padding='same', activation='relu'))
            self.encoder_layers.append(
                layers.MaxPooling2D(pool_size=(2, 2))
            )

        # Dense layer
        self.encoder_layers.append(layers.Flatten())
        self.encoder_layers.append(layers.Dense(code_length, activation='relu'))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    @property
    def last_dense_shape(self) -> int:
        return self.input_min_size // 2 ** len(self.conv_params)

    @property
    def last_conv_shape(self) -> Tuple[Union[int, Any], Union[int, Any], int]:
        dim1, dim2, channeldim = self.input_size
        dim_reduction_factor = 2 ** len(self.conv_params)
        return dim1 // dim_reduction_factor, dim2 // dim_reduction_factor, channeldim


class DecoderCNN(tf.keras.Model, ABC):
    def __init__(self, encoder: EncoderCNN):
        super(DecoderCNN, self).__init__()
        self.decoder_layers = []
        # Dense layer
        self.decoder_layers.append(layers.Dense(encoder.last_dense_shape, activation='relu'))
        self.decoder_layers.append(layers.Reshape(encoder.last_conv_shape))
        for param in reversed(encoder.conv_params):
            self.decoder_layers.append(layers.UpSampling2D(size=(2, 2)))
            self.decoder_layers.append(
                layers.Conv2DTranspose(kernel_size=param.kernel_size, filters=param.filters, padding='same'))
        self.decoder_layers.append(layers.Conv2D(kernel_size=1, filters=1))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.decoder_layers:
            x = layer(x)
        return x


class AutoEncoder:
    def __init__(self, input_size: Tuple[int, int, int], code_length: int):
        self.ae_input = layers.InputLayer(input_shape=input_size)
        self.encoder = EncoderCNN(input_size, code_length)
        self.decoder = DecoderCNN(self.encoder)
        self.model = tf.keras.Sequential(
            layers=[
                self.ae_input,
                self.encoder,
                self.decoder,
            ]
        )
        self.model.summary()


in_shape = (128, 128, 1)
code = 16

# ae = AutoEncoder(input_size=in_shape, code_length=16)
# ae.summary()

en = EncoderCNN(input_size=in_shape, code_length=code)
en.build(input_shape=in_shape)
en.summary()
