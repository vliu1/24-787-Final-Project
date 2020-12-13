# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:41:15 2020

@author: vince
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

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
data = pd.read_csv('data/{}.csv'.format(pollutant))

# Separate features and targets
X = data.values[:, 5:] # After 6th column
Y = data.values[:, label_index] # Labels

features_list = data.columns[5:]

# Normalize data
mean_x = np.mean(X, axis = 0)
std_dev = np.std(X, axis = 0)
X = (X-mean_x)/std_dev

mae_values = []
mse_values = []
r2_values = []

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

# Instantiate model with 1000 decision trees
n_estimators = 1000
rf = RandomForestRegressor(n_estimators = n_estimators).fit(X_train, y_train)

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2))
mae_values.append(round(np.mean(errors), 4))

# Calculate the squared errors
errors = (predictions - y_test)**2

# Print out the mean absolute error (mae)
#print('Mean Squared Error:', round(np.mean(errors), 2))
mse_values.append(round(np.mean(errors), 4))

r2_values.append(r2_score(y_test, predictions))

'''# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')'''


# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, features_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); 
plt.title('Variable Importances of {}, n_estimators = {}'.format(pollutant, n_estimators));