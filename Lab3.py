#Implement a Recurrent Neural Network (RNN)
import numpy as np

import tensorflow as  tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

mnist = tf.keras.datasets.mnist

#Split into Train and Test Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train[0].shape)

# setup RNN model
model = Sequential()

model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
model.fit(x_train, y_train, epochs=2)

model.evaluate(x_test, y_test)



