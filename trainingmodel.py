import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data() #input and output

cmap = 'Greys' #make images from dataset grey, easier to read 

x_train.shape
x_train = x_train.reshape(60000,28,28,1) #reshape to fit dataset
x_train = x_train / 255.0 #fixing scaling

x_test = x_test.reshape((10000,28,28,1))
y_test = y_test / 255.0

x_train.shape
x_test.shape

def CNN():
    inputs = keras.Input(shape=(28,28,1), name='Input layer')
    x = layers.Conv2D(filters=32, kernal_size = 3, strides = (1,1), padding='valid', activation='relu', name= "conv_layer_1")(inputs) #filter size 32, kernal 3 by 3, stride default is 1, activation function is relu
    x = layers.MaxPool2D(pool_size=(2, 2), name="pooling_1")(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name="conv_layer_2")(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name="conv_layer_3")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), name="pooling_2")(x)
    x = layers.Flatten(name="flattening_layer")(x) #Flatten layer to transform the data
    x = layers.Dense(units=64, activation="relu")(x) #Dense layers are fully connected,
    outputs = layers.Dense(units=10, activation='softmax', name='output_layer')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='first_CNN_model')
    model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

from keras.callbacks import ModelCheckpoint
modelcheckpoint = ModelCheckpoint(filepath="first_CNN.h5", save_best=True, monitor="val_loss")

history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, callbacks=[modelcheckpoint])
test_model = keras.models.load_model('first_CNN.h5')
test_model.evaluate(x_test, y_test)
