import tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #input and output

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x_train = x_train.reshape((60000, 28, 28, 1))/ 255.0
x_test = x_test.reshape((10000, 28, 28, 1))/ 255.0

def CNN():
    inputs = keras.Input(shape=(28, 28, 1), name='Input_layer')
    x = layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='valid', activation='relu', name="conv_layer_1")(inputs) #puts on 32 filters, sets kernel size to 3 x 3
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1")(x) #reduces the number of parameters in the input
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name="conv_layer_2")(x) #64 filters, 3 x 3 kernel
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2")(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name="conv_layer_3")(x) 
    x = layers.Flatten(name="flattening_layer")(x) #process of converting all of the multi-dimensional features into a one-dimensional vector
    x = layers.Dense(units=64, activation='relu')(x) #Dense layers are fully connected, meaning each neuron in the layer is connected to all the elements in the previous layer
    outputs = layers.Dense(units=10, activation='softmax', name='output_layer')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='first_CNN_model')
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

from tensorflow.keras.callbacks import ModelCheckpoint
modelcheckpoint = ModelCheckpoint(filepath="first_CNN.h5", save_best_only=True, monitor="val_loss")

history = model.fit(x=x_train, y=y_train,validation_data=(x_test, y_test),epochs=10, batch_size=64, callbacks=[modelcheckpoint])
