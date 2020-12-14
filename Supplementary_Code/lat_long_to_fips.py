# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:42:06 2020

@author: vince
"""


import requests
import urllib
import pandas as pd
import numpy as np

input_data = pd.read_csv("data/wind_farms.csv")

#Sample latitude and longitudes
#lat = [38.85210, 40.77720, 42.21240, 44.88200, 33.94250107, 	39.04880,29.99340 ]
#lon = [-77.0377, -73.8726, -83.3534,	-93.2218, -118.4079971, -84.6678, -90.2580]
lat = input_data["Latitude"]
lon = input_data["Longitude"]

fips_code = []
for i in range(len(lat)):
    latitude = lat[i]
    longitude = lon[i]
    #Encode parameters 
    params = urllib.parse.urlencode({'latitude': latitude, 'longitude':longitude, 'format':'json'})
    #Contruct request URL
    url = 'https://geo.fcc.gov/api/census/block/find?' + params
    
    #Get response from API
    response = requests.get(url)
    
    #Parse json in response
    data = response.json()
    
    #Print FIPS code
    fips_code.append(data['County']['FIPS'])

fips_code = np.array(fips_code)
fips_code = fips_code.reshape((len(fips_code),1))
results = fips_code
np.savetxt("output/wind_farms_fips.csv", results, delimiter=",", fmt = '%s')
