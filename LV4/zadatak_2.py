from sklearn import datasets
from sklearn.model_selection import train_test_split 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics
import pandas as pd
import math

from sklearn.metrics import max_error

from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('LV4\data_C02_emission.csv')
pd.set_option('display.max_columns', None)

var_input = ['Fuel Consumption City (L/100km)',
             'Fuel Consumption Hwy (L/100km)',
             'Fuel Consumption Comb (L/100km)',
             'Fuel Consumption Comb (mpg)',
             'Engine Size (L)',
             'Cylinders',
             'Fuel Type']

var_output = ['CO2 Emissions (g/km)']

ohe = OneHotEncoder ()
X_encoded=ohe.fit_transform(data[['Fuel Type']]).toarray()

labels=np.argmax(X_encoded,axis=1)

data['Fuel Type'] = labels

X=data[var_input]
y=data[var_output]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

linModel=lm.LinearRegression()
linModel.fit(X_train, y_train)
print(linModel.coef_)

y_test_p=linModel.predict(X_test)

MAE=sklearn.metrics.mean_absolute_error(y_test,y_test_p)
MSE=sklearn.metrics.mean_squared_error(y_test,y_test_p)
RMSE=math.sqrt(MSE)
MAPE=sklearn.metrics.mean_absolute_percentage_error(y_test,y_test_p)
R2=sklearn.metrics.r2_score(y_test,y_test_p)

# Dogodilo se povećanje regresijskih metrika
print('MAE:',MAE)
print('MSE:',MSE)
print('RMSE:',RMSE)
print('MAPE:',MAPE)
print('R2:',R2)

# Polje razlika izmedu CO2 emisija
errorArray = abs(y_test_p-y_test)
modelArray= data['Model'].to_numpy()

print('Najveća greška iznosi:',np.max(errorArray))
print('Auto asociran uz maksimalnu grešku je',modelArray[np.argmax(errorArray)])
