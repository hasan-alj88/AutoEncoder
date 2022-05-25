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
    """
    Get the offset mean absolute error of rounded true and prediction values from 100%.
    value 1.0 -> Zero mean absolute error
    :param y_true: true/Ground values
    :param y_pred: prediction values
    :return: rounded mean absolute error metric
    """
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
    """
    Get a random sample from tf.data.Dataset
    :param tf_ds: tf.data.Dataset object
    :param sample_size: sample size
    :return: random sample of x_data and y_data as numpy.array()
    """
    ds = tf_ds.shuffle(1024)
    x_data = np.array([x for x, _ in ds.take(sample_size)])
    y_data = np.array([y for _, y in ds.take(sample_size)])
    return x_data, y_data


def train_test_dataset_spilt(tf_ds: tf.data.Dataset,
                             split: float = 0.2,
                             batch_size: int = 32,
                             dataset_length: int = None) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Splits an Exsiting tf.Data.Dataset into Train and Test tf.data.Datasets
    :param tf_ds: The tf.Data.Dataset object
    :param split: test data fraction (0.0 < split < 1.0)
    :param batch_size: Batch size
    :param dataset_length: Defaults to len(tf_ds), otherwise can specify here.
    :return:
    """
    assert 0.0 < split < 1.0
    assert batch_size > 0
    ds = tf_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    n = len(ds) if dataset_length is None else dataset_length
    test_ds = ds.take(int(split * n)).shuffle(1024, reshuffle_each_iteration=True).prefetch(
        tf.data.experimental.AUTOTUNE)
    train_ds = ds.skip(int(split * n)).take(int((1 - split) * n)).shuffle(1024, reshuffle_each_iteration=True).prefetch(
        tf.data.experimental.AUTOTUNE)
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


def tensor_minmax_scaler(tensor):
    min_val = tf.math.reduce_min(tensor)
    max_val = tf.math.reduce_max(tensor)
    return (tensor - min_val) / max_val


def tensor_standard_scaler(tensor):
    mu = tf.math.reduce_mean(tensor)
    std = tf.math.reduce_std(tensor)
    return (tensor - mu) / std


def variance_outlier_extraction(tensor):
    tensor = tensor_minmax_scaler(tensor)
    t_row = relative_variance(tensor)
    t_col = relative_variance(tensor, axis=1)
    trc = tf.math.reduce_min(tf.concat([
        tf.expand_dims(t_row, axis=2),
        tf.expand_dims(t_col, axis=2),
    ],
        axis=2),
        axis=2)
    trc = tf.expand_dims(trc, axis=2)
    trc = tf.image.flip_up_down(trc)
    return tensor_minmax_scaler(tensor)


def variance_outlier_extraction_layer():
    return tf.keras.layers.Lambda(variance_outlier_extraction)


def time2vec(input_dim: int, output_dim: int, name: str = 'Time2Vec', **kwargs):
    """
    Time2Vec Encoding outputting Vector Representation of Time
    Citation:
    URL https://arxiv.org/abs/1907.05321
    @misc{https://doi.org/10.48550/arxiv.1907.05321,
    doi = {10.48550/ARXIV.1907.05321},
    url = {https://arxiv.org/abs/1907.05321},
    author = {Kazemi, Seyed Mehran and Goel, Rishab and Eghbali, Sepehr and Ramanan, Janahan and Sahota,
    Jaspreet and Thakur, Sanjay and Wu, Stella and Smyth, Cathal and Poupart, Pascal and Brubaker, Marcus},
    keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Time2Vec: Learning a Vector Representation of Time},
    publisher = {arXiv},
    year = {2019},
    copyright = {Creative Commons Attribution Non-Commercial Share Alike 4.0 International}
    }
    :param input_dim: Size of the input (batch, input_dim)
    :param output_dim: Size of the output (batch, output_dim)
    :param name: Keras Layer name
    :param kwargs: tf.keras.Model() keywords
    :return: (batch, output_dim) Vector Representation of Time
    """
    # tou (batch, signal)
    # y0 = w0 . t + phi0 ; k=0
    tou = tf.keras.layers.Input((input_dim,))
    y0 = tf.keras.layers.Dense(1, activation=None)(tou)
    # y = sin( W . t + Phi ); 0<k<input_dim
    y = tf.keras.layers.Dense(output_dim - 1, activation=tf.math.sin)(tou)
    t2vec_code = tf.keras.layers.Concatenate(axis=1)([y0, y])
    return tf.keras.Model(tou, t2vec_code, name=name, **kwargs)


def euclidean_distance_layer(input_dim: int, name: str = 'Euclidean_Distance', **kwargs):
    """
    Merging keras layer where it output the Euclidean Distance of two 1D tensors
    :param input_dim: the two tensors dimension size (batch, input_dim)
    :param kwargs: tf.keras.Model() keywords
    :param name: Keras Layer name
    :return: The Euclidean Distance of two 1D tensors (batch, input_dim)
    """
    xin = tf.keras.Input((input_dim,))
    x_sqr = tf.keras.layers.Lambda(tf.math.square, name='Square')(xin)
    x_sqr_sum = tf.keras.layers.Lambda(tf.math.reduce_sum,
                                       name='sqr_sum',
                                       arguments={'axis': 1})(x_sqr)
    distance = tf.keras.layers.Lambda(tf.math.sqrt)(x_sqr_sum)
    return tf.keras.Model(xin, distance, name=name, **kwargs)
