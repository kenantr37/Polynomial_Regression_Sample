# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:35:30 2020

@author: Zeno
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # we imported it for comparing polynomial and linear regression
from sklearn.preprocessing import PolynomialFeatures # we imported this for using polynomial regression 
"""
in this regression , we want to predict car's speed which depends car's price
we want to see stable and balanced predictions and for this we used polynomial
regression here.In the end , we will compare what would happen what if we would
use linear regression for this dataset.
        
"""
data = pd.read_csv("D:/Machine Learning Works/Polynomial Linear Regression/polynomialregression.csv",sep =";")#reading 
x = data.iloc[:,0].values.reshape(-1,1) # price of the car
y = data.speed.values #speed of the car 

polynomial_sample = PolynomialFeatures(degree = 5).fit_transform(x) # we need to fit and transform x for make x^n (as I understood)
linear_sample_for_line = LinearRegression().fit(polynomial_sample, y)# we fit polynomial_sample and y for predict x 
y_head = linear_sample_for_line.predict(polynomial_sample)  # now our prediction is ready to use
linear_and_polynomial_comparing = LinearRegression().fit(x,y) #for comparing

#for comparing linear regression and polynomial regression we can create linear model
y_head_linear = linear_and_polynomial_comparing.predict(x) # as you can see, we did not put polynomial_sample because we don't want x^n

#now we can attempt to visualize our prediction
plt.scatter(x,y,color = "purple",alpha=(0.3))
plt.plot(x,y_head,color = "brown",label = "Polynomial Regression model")
plt.plot(x,y_head_linear,color = "red",label ="Linear Regression model for comparing")
plt.xlabel("price of the car")
plt.ylabel("speed of the car")
plt.legend()
plt.show()