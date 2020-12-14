#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
data = pd.read_csv('data/all-airport-data.csv')
zip_codes_dataset = pd.read_csv('data/ZIP-COUNTY-FIPS_2018-03.csv')




zip_code = data['ZipCode']
zip_code = zip_code.fillna(0)
zip_code = np.array(zip_code)
zip_code_dataset = zip_codes_dataset['ZIP']
fips_dataset = zip_codes_dataset['STCOUNTYFP']

county = data['County']
county_dataset = zip_codes_dataset['COUNTYNAME']


fips_codes = np.zeros(len(zip_code))
for i in range(len(fips_codes)):
    #print(i, zip_code[i], fips_codes[i])
    if zip_code[i] == 0:
        fips_codes[i] = None
        continue
    if zip_code[i].isnumeric():
        mask = (int(zip_code[i]) == zip_code_dataset)
        if (sum(mask) == 0):
            fips_codes[i] = None
            continue
        fips_codes[i] = int(fips_dataset[mask].values[0])
    else:
        fips_codes[i] = None
    
          
fips_codes = fips_codes.reshape((len(fips_codes),1))
np.savetxt("output/all-airport-data_fips.csv".format(column_name), fips_codes, delimiter=",", fmt = '%s')
#print(fips_codes)


