import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Set colormap for plotting images (if needed)
cmap = 'Greys'

# Reshape and normalize the data
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

print(x_train.shape)  # For debugging purposes
print(x_test.shape)   # For debugging purposes

# Define the CNN model
def CNN():
    inputs = keras.Input(shape=(28, 28, 1), name='Input_layer')
    x = layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='valid', activation='relu', name="conv_layer_1")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1")(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name="conv_layer_2")(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name="conv_layer_3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2")(x)
    x = layers.Flatten(name="flattening_layer")(x)
    x = layers.Dense(units=64, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax', name='output_layer')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='first_CNN_model')
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate and train the model
model = CNN()

from tensorflow.keras.callbacks import ModelCheckpoint
modelcheckpoint = ModelCheckpoint(filepath="first_CNN.h5", save_best_only=True, monitor="val_loss", mode='min')

history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, callbacks=[modelcheckpoint])

# Load the best model and evaluate
test_model = keras.models.load_model('first_CNN.h5')
test_model.evaluate(x_test, y_test)

model.save('cnn_model.h5')
