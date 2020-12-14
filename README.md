# 24-787-Final-Project
Using infrastructure data to predict pollution levels

Above is the code and data for my group's 24-787 Machine Learning and Artificial Intelligence for Engineers final project. This class was taken during my Senior Fall 2020 year. 

Packages Used:
- Tensorflow
- Keras
- Sklearn
- Numpy
- Pandas


Abstract: 
An attempt to build models that predict pollution levels of a US county based on infrastructure features has been done in this study. US EPA maintains the database of pollution levels of various pollutants out of which we have used 4 pollution data: PM2.5, SO2, Ozone, NO2. 15 infrastructural features in each county have been used to train various Supervised Learning algorithms. These features include Coal Plants, Airports, Public Schools, Hospitals, Fuel Stations, Land Area, Population etc. It was noted that the MSE errors for the regression models was very high and are not suitable for real life application. Furthermore, pollution which was originally in numerical form was converted to categorical data and classification model was carried out. This model gave very good results for PM2.5 (75%) accuracy, while returning <45% accuracy for SO2, Ozone, NO2. It is thus noted that the infrastructure data used by us is inadequate to predict levels of PM2.5, SO2, Ozone, NO2 within comfortable margin of error. Classification model may better serve our purpose but only for PM2.5, the accuracy for other pollutants is subpar. In conclusion, infrastructure data is not a good direct indicator of pollution but can be used to indirectly predict sources of pollution. We further discuss the limitations and possible future improvements to the model.
