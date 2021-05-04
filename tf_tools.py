from abc import ABC
from tensorflow.keras import Sequential, layers, Model, losses
import numpy as np
import tensorflow as tf
import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List


def tf_dataset_itr(tf_ds: tf.data.Dataset):
    for x_batch, y_batch in tf_ds:
        for x, y in zip(x_batch, y_batch):
            yield x, y


def rounded_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


def encoder_decoder_layers(units: int,
                           input_shape: Tuple,
                           layer_type: str = 'Dense',
                           filters: Optional[int] = None,  # default is 4
                           pool: Optional[int] = None):  # default is pooling is not applied
    if not layer_type == 'convolution':
        return tf.keras.layers.Dense(units=units), tf.keras.layers.Dense(units=units)
    else:
        filters = 4 if filters is None else filters
        xin = tf.keras.layers.InputLayer(input_shape)
        x = layers.Conv2D(units, (filters, filters), activation='relu', padding='same')(xin)
        if pool is not None:
            encoded = layers.MaxPooling2D((pool, pool), padding='same')(x)
        else:
            encoded = x
        x = layers.Conv2D(units, (filters, filters), activation='relu', padding='same')(encoded)
        if pool is None:
            decoded = layers.UpSampling2D((pool, pool))(x)
        else:
            decoded = x
        return tf.keras.Model(xin, encoded), tf.keras.Model(encoded, decoded)


def plot_confusion_matrix(model: tf.keras.Model,
                          dataset: tf.data.Dataset,
                          prediction_function: callable = np.argmax):
    ds = dataset.unbatch()
    y_true = np.array([y for _, y in ds.unbatch().as_numpy_iterator()])
    y_pred = [model.predict(x) for x, _ in ds.unbatch().as_numpy_iterator()]
    y_pred = np.array(map(prediction_function, y_pred))
    cm = tf.math.confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(np.array(cm))
    sns.heatmap(cm, annot=True)


def get_x_shape(tf_ds):
    return [x for x, _ in tf_ds.take(1)][0].numpy().shape


def ds_x_data(tf_ds):
    return np.array([x for x, _ in tf_ds.as_numpy_iterator()])


def ds_y_data(tf_ds):
    return np.array([y for _, y in tf_ds.as_numpy_iterator()])


def get_random_sample(tf_ds: tf.data.Dataset,
                      sample_size: int = 1):
    ds = tf_ds.shuffle(1024)
    x_data = np.array([x for x, _ in ds.take(sample_size)])
    y_data = np.array([y for _, y in ds.take(sample_size)])
    return x_data, y_data


def train_test_dataset_spilt(tf_ds: tf.data.Dataset,
                             split: float = 0.2,
                             batch_size: int = 32, ) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    assert 0.0 < split < 1.0
    assert batch_size > 0
    ds = tf_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    n = len(ds)
    test_ds = ds.take(int(split * n))
    train_ds = ds.skip(int(split * n)).take(int((1 - split) * n))
    return train_ds, test_ds


class AutoEncoder(Model):
    def __init__(self, tf_ds: tf.data.Dataset,
                 units: List[int],
                 layers_type: str = 'Convolutional',
                 filters: Optional[List[int]] = None,
                 pools: Optional[List[int]] = None):
        tf_ds = tf_ds.shuffle(2048)
        x_data = ds_x_data(tf_ds)
        x_data = x_data / np.max(x_data)  # Normalize
        ic(x_data.shape)
        self.ds = tf.data.Dataset.from_tensor_slices((x_data, x_data))
        self.ds = self.ds.shuffle(1024).batch(100).prefetch(tf.data.experimental.AUTOTUNE)
        ic(self.ds)
        self.train_ds, self.val_ds = train_test_dataset_spilt(self.ds)
        ic(self.train_ds)
        ic(self.val_ds)

        self.AE_input_shape = get_x_shape(self.train_ds)[1:]
        ic(self.AE_input_shape)

        x_in = layers.Input(self.AE_input_shape)
        ic(x_in)
        x = x_in
        decoder_layers = []
        for Unit, Filter, Pool in zip(units, filters, pools):
            encode_layer, decode_layer = encoder_decoder_layers(
                units=Unit,
                filters=Filter,
                pool=Pool,
                layer_type=layers_type,
                input_shape=x.shape,
            )
            decoder_layers.append(decode_layer)
            x = encode_layer(x)
            ic(decode_layer)
            ic(encode_layer)
            ic(x)

        self.code = x
        ic(self.code)
        for decode_layer in reversed(decoder_layers):
            x = decode_layer(x)
            ic(x)
        else:
            decoded = x

        super(AutoEncoder, self).__init__(x_in, decoded)
        self.compile(optimizer=tf.keras.optimizers.SGD(0.001),
                     loss=tf.keras.losses.BinaryCrossentropy(),
                     metrics=[rounded_accuracy],
                     )
        self.summary()
        self.History = self.fit(
            self.train_ds,
            epochs=512,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3,
                                                 verbose=1,
                                                 restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='rounded_accuracy',
                                                     factor=0.5,
                                                     patience=1,
                                                     verbose=1),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
            verbose=1,
            validation_data=self.val_ds,
        )

    def plot_loss(self):
        ic(self.History)
        plt.plot(self.History.history['val_loss'], label='val_loss')
        plt.plot(self.History.history['loss'], label='loss')
        plt.title('AutoEncoder Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()
