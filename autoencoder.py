from math import e
import re
from tkinter import N
import tensorflow as tf
from keras import layers, Sequential
from functools import reduce
from icecream import ic
from collections import namedtuple


def output_flat_size(model):
    return reduce(lambda x,y:x*y,model.layers[-1].output_shape[1:])

def min_hight_output(model):
    return min(model.layers[-1].output_shape[1:-1])


class AutoEncoder(tf.keras.Model):
    def __init__(self,input_height:int,input_width:int,coder_size:int,channels:int=1, batch_size:int=10):
        self.encoder = Sequential(name='Encoder')
        input_shape = (input_height,input_height,channels)
        self.encoder.add(layers.InputLayer(input_shape=input_shape, batch_size=batch_size) )
        self.encoder.add(layers.BatchNormalization())
        self.convlution_layers_count = 0
        base_kernel_size = coder_size 
        self.conv_params = []
        conv_params = namedtuple('conv_params',['kernel_size','filters'])
        while True:
            try:
                if output_flat_size(self.encoder) > coder_size * 2:
                    f = min_hight_output(self.encoder) // 3
                    self.encoder.add(layers.Conv2D(kernel_size=base_kernel_size,filters=f, padding='same' , activation='relu'))
                    self.encoder.add(layers.MaxPooling2D(pool_size=(2,2)))
                    self.conv_params.append(conv_params(kernel_size=base_kernel_size, filters=f))
                    self.convlution_layers_count += 1       
                else:
                    break
            except ValueError:
                break
        last_conv_shape = self.encoder.layers[-1].output_shape[1:]
        self.encoder.add(layers.Flatten())
        last_dense_shape = self.encoder.layers[-1].output_shape[1]
        self.encoder.add(layers.Dense(coder_size, activation='relu'))
        self.encoder.build(input_shape=(input_height,input_height,channels))
        self.encoder.summary()
        
        
        self.decoder = Sequential(name='Decoder')
        self.decoder.add(layers.InputLayer(input_shape=(coder_size,), batch_size=batch_size))
        self.decoder.add(layers.Dense(last_dense_shape, activation='relu'))
        self.decoder.add(layers.Reshape(last_conv_shape))
        for param in reversed(self.conv_params):
            self.decoder.add(layers.UpSampling2D(size=(2,2)))
            self.decoder.add(layers.Conv2DTranspose(kernel_size=param.kernel_size,filters=param.filters, padding='same'))
        self.decoder.add(layers.Conv2D(kernel_size=1, filters=1))
        self.decoder.build(input_shape=(coder_size,))
        self.decoder.summary()
        ic(input_shape)
        input = tf.keras.Input(shape=input_shape, batch_size=batch_size)
        code = self.encoder(input)
        output = self.decoder(code)
        super(AutoEncoder,self).__init__(input, output)
        self.summary()

ae = AutoEncoder(128,64,16)
        


