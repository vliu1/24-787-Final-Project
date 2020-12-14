# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:18:52 2020

@author: vince
"""


import pandas as pd
import numpy as np

data = pd.read_csv('data/Overall_data.csv')

column_name = "Airports"

feature = data[column_name]
fips_codes = data["Fips_code"]

# Replaces nan and None values to 0
feature = feature.fillna(0)
feature = feature.replace(to_replace = 'None', value = 0)

# Convert array values to int
feature = np.array(feature).astype(int)
fips_codes = np.array(fips_codes).astype(int)

results = np.zeros(len(fips_codes))
for i in range(len(fips_codes)):
    if fips_codes[i] == 0:
        fips_codes[i] = None
    else:
        mask = (fips_codes[i] == feature)
        results[i] = sum(mask)

results = results.reshape((len(results),1))
np.savetxt("output/{}_fips.csv".format(column_name), results, delimiter=",", fmt = '%s')