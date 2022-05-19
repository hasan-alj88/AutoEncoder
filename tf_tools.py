from abc import ABC
from tensorflow.keras import Sequential, layers, Model, losses
import numpy as np
import tensorflow as tf
import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from tensorflow.python.keras.layers import Lambda


def tf_dataset_itr(tf_ds: tf.data.Dataset):
    for x_batch, y_batch in tf_ds:
        for x, y in zip(x_batch, y_batch):
            yield x, y


def rounded_accuracy(y_true, y_pred):
    return 1 - tf.keras.metrics.mean_absolute_error(tf.round(y_true), tf.round(y_pred))


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
    return np.squeeze(next(tf_dataset_itr(tf_ds))[0]).shape


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


def relative_tensor(tensor: tf.float64, row: int):
    return tf.concat([tensor[:row], tensor[row + 1:]], axis=0)


def relative_variance(tensor: tf.float64, axis: int = 0):
    if axis > 2:
        raise ValueError('axis must be 0->rows or 1->columns.')
    elif axis == 1:
        t = tf.transpose(tensor)
    else:
        t = tensor
    trr_rows = []
    for row in range(tensor.shape[0]):
        trr_row = relative_tensor(t, row)
        trr_row = tf.math.reduce_variance(trr_row, axis=1)
        trr_row = tf.expand_dims(trr_row, axis=1)
        trr_rows.append(trr_row)
    trr = tf.concat(trr_rows, axis=1)
    trr = tf.squeeze(trr)
    return tf.transpose(trr) if axis == 1 else trr


def variance_outlier_extraction(tensor):
    t_row = relative_variance(tensor)
    t_col = relative_variance(tensor, axis=1)
    trc = tf.math.reduce_min(tf.concat([
        tf.expand_dims(t_row, axis=2),
        tf.expand_dims(t_col, axis=2),
    ],
        axis=2),
        axis=2)
    trc = tf.expand_dims(trc, axis=2)
    return tf.image.flip_up_down(trc)


class variance_outlier_extraction_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(variance_outlier_extraction_layer, self).__init__()

    def call(self, inputs, **kwargs):
        return tf.keras.layers.Lambda(variance_outlier_extraction)(inputs)
