# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:45:07 2020

@author: vince
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time


pollutant = "NO2_Classification"

if pollutant == "PM2.5_Classification":
    label_index = 1
if pollutant == "SO2_Classification":
    label_index = 2
if pollutant == "Ozone_Classification":
    label_index = 3
if pollutant == "NO2_Classification":
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

# Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

accuracy_values = []

# Cross-Validation
kf5 = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kf5.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Instantiate model with 1000 decision trees
    n_estimators = 5000
    rf = RandomForestClassifier(n_estimators = n_estimators, random_state = 1).fit(X_train, y_train)
    
    # Use the forest's predict method on the test data
    predictions = rf.predict(X_test)
    
    # Check Model Accuracy
    mask = (y_test == predictions)
    accuracy = sum(mask)/y_test.shape[0]
    accuracy_values.append(accuracy)

print("Test Accuracy:", np.mean(accuracy_values))

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