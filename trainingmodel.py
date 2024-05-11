import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Checking GPU availability
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Downloading the dataset for training
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

X_train = np.expand_dims(X_train, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Declaring the CNN model
model = models.Sequential([
    layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu', strides=1, input_shape=[28,28,1]),
    layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1, activation='relu'),
    layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    layers.Flatten(),
    layers.Dense(units=10, activation='relu'),
    layers.Dense(units=10, activation='softmax')
])

# Selecting an optimizer
opt = optimizers.SGD(learning_rate=0.01)

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Training the network
history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_valid, y_valid))

# Extracting the weights from the network
for layer in model.layers:
    if len(layer.weights) > 0:
        if len(layer.weights[0].shape) == 4:
            converted_weights = np.transpose(layer.weights[0], [3, 2, 0, 1])
            np.savetxt(layer.name + ".txt", converted_weights.flatten(), delimiter=",")
        else:
            converted_weights = np.transpose(layer.weights[0], [1,0])
            np.savetxt(layer.name + ".txt", converted_weights.flatten(), delimiter=",")
