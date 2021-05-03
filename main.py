import matplotlib.pyplot as plt
import numpy as np

import tf_tools
from tf_tools import AutoEncoder
import tensorflow as tf
import tensorflow_datasets as tfds
from icecream import ic

# Construct a tf.data.Dataset
ds = tfds.load('mnist', split='train', shuffle_files=True, as_supervised=True)

# Build your input pipeline
ds = ds.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)
ic(ds)
ae = AutoEncoder(ds)
print(ae)

x, y = tf_tools.get_random_sample(ds)
plt.subplot(211)
plt.imshow(np.squeeze(x)/np.max(x))
plt.subplot(212)
plt.imshow(np.squeeze(ae(x)))
plt.show()
ae.plot_loss()