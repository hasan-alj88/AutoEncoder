import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# from icecream import ic
from typing import Tuple


class AutoEncoder:
    def __init__(self, input_size: Tuple[int, int, int], code_length: int):
        self.code_length = code_length
        self.input_size = input_size
        self.encoder = None
        self.decoder = None
        self.conv_params = None
        self.last_conv_shape = None
        self.first_dense_shape = None

        self.ae_input = layers.InputLayer(input_shape=input_size)
        self.create_encoder()
        self.create_decoder()
        self.model = tf.keras.Sequential(
            layers=[
                self.ae_input,
                self.encoder,
                self.decoder,
            ],
            name='AutoEncoder'
        )
        self.model.summary()

    def create_encoder(self) -> None:
        self.encoder = tf.keras.Sequential(name='Encoder')
        self.encoder.add(layers.InputLayer(input_shape=self.input_size))
        min_input_dim = min(self.input_size[:-1])
        number_of_conv_layers = int(np.log2(min_input_dim))
        self.conv_params = [(min_input_dim // 2 ** layer_ind, 3)
                            for layer_ind in range(number_of_conv_layers)
                            if min_input_dim // 2 ** layer_ind * 3 > self.code_length * 2]
        for f, k in self.conv_params:
            self.encoder.add(layers.Conv2D(filters=f, kernel_size=k, padding='same'))
            self.encoder.add(layers.MaxPooling2D(pool_size=(2, 2)))
            self.encoder.add(layers.ReLU())
        self.last_conv_shape = self.encoder.layers[-1].output_shape

        self.encoder.add(layers.Flatten())
        self.first_dense_shape = self.encoder.layers[-1].output_shape[1]
        self.encoder.add(layers.Dense(code, activation='relu'))
        self.encoder.summary()

    def create_decoder(self) -> None:
        self.decoder = tf.keras.Sequential(name='Decoder')
        self.decoder.add(layers.InputLayer(input_shape=(self.code_length,)))
        self.decoder.add(layers.Dense(self.first_dense_shape, activation='relu'))
        self.decoder.add(layers.Reshape(target_shape=self.last_conv_shape[1:]))
        for f, k in reversed(self.conv_params):
            self.decoder.add(layers.Conv2DTranspose(filters=f, kernel_size=k, padding='same'))
            self.decoder.add(layers.UpSampling2D((2, 2)))
            self.decoder.add(layers.ReLU())
        self.decoder.add(layers.Conv2D(filters=1, kernel_size=1, padding='same'))
        self.decoder.summary()


in_shape = (128, 64, 1)
code = 16

ae = AutoEncoder(input_size=in_shape, code_length=16)
# ae.summary()
