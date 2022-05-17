import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from enum import Enum
# from icecream import ic
from typing import Tuple


class AutoEncoder_type(Enum):
    Convolutional2D = 0
    DenseLayered = 1
    Convolutional1D = 2


class AutoEncoder:
    def __init__(self, input_size: Tuple[int, int, int],
                 code_length: int,
                 ae_kind: AutoEncoder_type = AutoEncoder_type.Convolutional2D):
        self.code_length = code_length
        self.input_size = input_size
        self.encoder = None
        self.decoder = None
        self.conv_params = None
        self.last_conv_shape = None
        self.first_dense_shape = None
        self.model = None
        self.min_input_dim = min(self.input_size[:-1])
        autoencoder_creation_methods = {
            AutoEncoder_type.Convolutional2D: self.convolutional_autoencoder,
            AutoEncoder_type.DenseLayered: self.dense_layered_autoencoder}
        autoencoder_creation_methods[ae_kind]()

    def convolutional_autoencoder(self) -> None:
        ae_input = layers.InputLayer(input_shape=self.input_size)
        self.create_encoder()
        self.create_decoder()
        self.model = tf.keras.Sequential(
            layers=[
                ae_input,
                self.encoder,
                self.decoder,
            ],
            name='AutoEncoder'
        )
        self.model.summary()

    def create_encoder(self) -> None:
        self.encoder = tf.keras.Sequential(name='Encoder')
        self.encoder.add(layers.InputLayer(input_shape=self.input_size))

        number_of_conv_layers = int(np.log2(self.min_input_dim))
        self.conv_params = [(self.min_input_dim // 2 ** layer_ind, 3)
                            for layer_ind in range(number_of_conv_layers)
                            if self.min_input_dim // 2 ** layer_ind * 3 > self.code_length * 2]
        for f, k in self.conv_params:
            self.encoder.add(layers.Conv2D(filters=f, kernel_size=k, padding='same'))
            self.encoder.add(layers.MaxPooling2D(pool_size=(2, 2)))
            self.encoder.add(layers.ReLU())
        self.last_conv_shape = self.encoder.layers[-1].output_shape

        self.encoder.add(layers.Flatten())
        self.first_dense_shape = self.encoder.layers[-1].output_shape[1]
        self.encoder.add(layers.Dense(code, activation=None))
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

    def dense_layered_autoencoder(self):

        self.encoder = tf.keras.Sequential(name='Encoder')
        self.encoder.add(layers.InputLayer(input_shape=self.input_size))
        self.encoder.add(layers.Flatten())
        self.encoder.add(layers.BatchNormalization())
        self.encoder.add(layers.Flatten())
        dense_length = self.encoder.output_shape[1]
        dense_layers = [dense_length // 2 ** dl for dl in range(int(np.log2(dense_length)))
                        if dense_length // 2 ** dl > self.code_length * 2]
        for dense_layer in dense_layers:
            self.encoder.add(layers.Dense(dense_layer, activation='relu'))
        self.encoder.add(layers.Dense(self.code_length, activation=None))

        self.decoder = tf.keras.Sequential(name='Decoder')
        self.decoder.add(layers.InputLayer(input_shape=(self.code_length,)))
        for dense_layer in reversed(dense_layers):
            self.decoder.add(layers.Dense(dense_layer))
        self.decoder.add(layers.Reshape(self.input_size))

        self.model = tf.keras.Sequential(name='AutoEncoder')
        self.model.add(layers.InputLayer(input_shape=self.input_size))
        self.model.add(self.encoder)
        self.model.add(self.decoder)

    def convolutional1d_auto_encoder(self) -> None:

        self.encoder = tf.keras.Sequential(name='Encoder')
        self.encoder.add(layers.InputLayer(input_shape=self.input_size))
        self.encoder.add(layers.Flatten())
        self.encoder.add(layers.BatchNormalization())
        self.encoder.add(layers.Flatten())
        conv1d_length = self.encoder.output_shape[1]
        conv1d_layers = [conv1d_length // 2 ** dl for dl in range(int(np.log2(conv1d_length)))
                         if conv1d_length // 2 ** dl > self.code_length * 2]
        for conv1d_layer in conv1d_layers:
            self.encoder.add(layers.Conv1D(
                kernel_size=conv1d_layer, filters=3, padding='same', activation='relu'))
            self.encoder.add(layers.MaxPooling1D(pool_size=2))
        self.encoder.add(layers.Dense(self.code_length, activation=None))

        self.decoder = tf.keras.Sequential(name='Decoder')
        self.decoder.add(layers.InputLayer(input_shape=(self.code_length,)))
        for conv1d_layer in reversed(conv1d_layers):
            self.decoder.add(layers.Conv1DTranspose(
                kernel_size=conv1d_layer, filters=3, padding='same', activation='relu'))
            self.decoder.add(layers.UpSampling1D(size=2))
        self.decoder.add(layers.Reshape(self.input_size))

        self.model = tf.keras.Sequential(name='AutoEncoder')
        self.model.add(layers.InputLayer(input_shape=self.input_size))
        self.model.add(self.encoder)
        self.model.add(self.decoder)


in_shape = (128, 64, 1)
code = 16

ae = AutoEncoder(input_size=in_shape, code_length=16)
# ae.summary()
