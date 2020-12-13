# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:27:06 2020

@author: vince
"""


# Load dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import pandas as pd
import time

pollutant = "SO2"

if pollutant == "PM2.5":
    label_index = 1
if pollutant == "SO2":
    label_index = 2
if pollutant == "Ozone_8_hour":
    label_index = 3
if pollutant == "NO2_1_hour":
    label_index = 4

# Load data
data = pd.read_csv('data/{}.csv'.format(pollutant)).values

# Separate features and targets
X = data[:, 5:] # After 6th column
Y = data[:, label_index] # Second column

# Normalize data
mean_x = np.mean(X, axis = 0)
std_dev = np.std(X, axis = 0)
X = (X-mean_x)/std_dev

mae_values = []
mse_values = []
#for i in range(50):
# Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


t = time.time()
mse_values = []
mae_values = []
r2_values = []
# Cross-Validation
kf5 = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kf5.split(X):
    # Split train-test based on k-fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # Set the input shape
    input_shape = (X.shape[1], )
    
    # Model building
    model = Sequential()
    model.add(Dense(50, input_shape=input_shape, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    # Run model
    t = time.time()
    optimizer = tensorflow.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    model.fit(X_train, y_train, epochs=1000, validation_split=0.2)
    
    # Score model
    score = model.evaluate(X_test, y_test, verbose=1)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    
    mse_values.append(score[0])
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    mae_values.append(round(np.mean(errors), 4))
    
    # Calculate R^2
    r2_values.append(r2_score(y_test, predictions))
    

# Print out the mse
print('Mean Squared Error:', np.mean(mse_values))

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', np.mean(mae_values))
    
elapsed = time.time() - t  
print("Elapsed Time:", elapsed)

print('R^2 Score:', np.mean(r2_values))


