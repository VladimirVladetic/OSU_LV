from sklearn import datasets
from sklearn.model_selection import train_test_split 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics
import pandas as pd
import math
from sklearn.preprocessing import OneHotEncoder

data=pd.read_csv('data_C02_emission.csv')

ohe=OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()

var_input = ['Fuel Consumption City (L/100km)',
             'Fuel Consumption Hwy (L/100km)',
             'Fuel Consumption Comb (L/100km)',
             'Fuel Consumption Comb (mpg)',
             'Engine Size (L)',
             'Cylinders',
             'Fuel Type']

y=data[['CO2 Emissions (g/km)']]
X=data[var_input]
X['Fuel Type'] = X_encoded
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

linModel = lm.LinearRegression()
linModel.fit(X_train, y_train)
print(linModel.coef_)

y_test_p = linModel.predict(X_test)

plt.scatter(x=X_test['Fuel Consumption City (L/100km)'],y=y_test,s=10,c='b')
plt.scatter(x=X_test['Fuel Consumption City (L/100km)'],y=y_test_p,s=10,c ='r')
plt.show()