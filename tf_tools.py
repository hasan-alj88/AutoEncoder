from abc import ABC
from tensorflow.keras import Sequential, layers, Model, losses
import numpy as np
import tensorflow as tf


def tf_dataset_itr(tf_ds: tf.data.Dataset):
    for x_batch, y_batch in tf_ds:
        for x, y in zip(x_batch, y_batch):
            yield x, y


class AutoEncoder(Model, ABC):
    def __init__(self, tf_ds: tf.data.Dataset):
        super(AutoEncoder, self).__init__()
        self.data = np.array([x for x, y in tf_dataset_itr(tf_ds)])
        self.input_dim = len(self.data.shape)
        self.AE_input_shape = self.data[0].shape
        self.AE_input_shape = np.squeeze(self.input_dim).shape + (1,)
        self.noise = Sequential([
            layers.LayerNormalization(),
            layers.GaussianNoise()
        ])

        self.encoder = Sequential([
            layers.Input(shape=self.AE_input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
        ])

        self.decoder = Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

        self.compile(optimizer='adam', loss=losses.MeanSquaredError())
        self.fit(x=self.data, y=self.data)

    def call(self, x):
        noisy_x = self.noise(x)
        encoded = self.encoder(noisy_x)
        decoded = self.decoder(encoded)
        return decoded
