# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:11:17 2020

@author: vince
"""


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import pandas as pd
import time
import numpy as np

# Data preprocessing
num_classes = 4

pollutant = "Ozone_Classification"

if pollutant == "PM2.5_Classification":
    label_index = 1
if pollutant == "SO2_Classification":
    label_index = 2
if pollutant == "Ozone_Classification":
    label_index = 3
if pollutant == "NO2_Classification":
    label_index = 4

# Load data
data = pd.read_csv('data/{}.csv'.format(pollutant)).values
X = data[:, 5:] # After 6th column
Y = data[:, label_index] # Second column
Y = Y - 1

# Normalize data
mean_x = np.mean(X, axis = 0)
std_dev = np.std(X, axis = 0)
X = (X-mean_x)/std_dev

# Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# One-hot Vector
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

t = time.time()
loss_values = []
accuracy_values = []
# Cross-Validation
kf5 = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kf5.split(X):
    # Split train-test based on k-fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # One-hot vector
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # Building Model
    model = Sequential()
    
    model.add(Dense(50, input_dim = X.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
    
    model.fit(X_train, y_train, epochs = 1000)
    
    # Model Accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    loss_values.append(loss)
    accuracy_values.append(accuracy)

elapsed_time = time.time() - t

print("Elapsed Time =", elapsed_time)
print("Test Loss:", np.mean(loss_values))
print("Test Accuracy:", np.mean(accuracy_values))