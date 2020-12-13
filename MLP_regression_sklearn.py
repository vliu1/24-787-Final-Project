# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:57:32 2020

@author: vince
"""


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np

import pandas as pd
import time

pollutant = "NO2_1_hour"

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

# Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Hyperparameter Tuning
mlp = MLPRegressor()

param_grid = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,1)],
          'activation': ['relu'],
          'alpha': [0.0001, 0.05],
          'learning_rate': ['constant','adaptive'],
          'solver': ['adam']}

gsc = GridSearchCV(
    mlp,
    param_grid,
    cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

grid_result = gsc.fit(X, Y)

best_params = grid_result.best_params_

# Model 
t = time.time()
best_mlp = MLPRegressor(hidden_layer_sizes = best_params["hidden_layer_sizes"], 
                        activation =best_params["activation"],
                        solver=best_params["solver"],
                        max_iter= 1000, n_iter_no_change = 200
              ).fit(X_train, y_train)


predictions = best_mlp.predict(X_test)
errors = (predictions - y_test)**2
print('Mean Squared Error:', round(np.mean(errors), 4))
print("Score:", best_mlp.score(X_test, y_test))

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 4))

elapsed = time.time() - t
print("Elapsed Time:", elapsed)




