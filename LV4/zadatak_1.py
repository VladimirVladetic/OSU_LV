from sklearn import datasets
from sklearn.model_selection import train_test_split 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics
import pandas as pd
import math

from sklearn.preprocessing import StandardScaler 

data = pd.read_csv('data_C02_emission.csv')

var_input = ['Fuel Consumption City (L/100km)',
             'Fuel Consumption Hwy (L/100km)',
             'Fuel Consumption Comb (L/100km)',
             'Fuel Consumption Comb (mpg)',
             'Engine Size (L)',
             'Cylinders']

var_output = ['CO2 Emissions (g/km)']

X=data[var_input].to_numpy()
y=data[var_output].to_numpy()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state=1)

plt.scatter(x=np.transpose(X_train[:,0]),y=y_train,c='b',s=10)
plt.scatter(x=np.transpose(X_test[:,0]),y=y_test,c='r',s=10)
plt.show()

ss = StandardScaler()
X_train_n=ss.fit_transform(X_train)
X_test_n=ss.transform(X_test)

plt.figure
plt.hist(np.transpose(X_train[:,0]))
plt.show()
plt.figure
plt.hist(np.transpose(X_train_n[:,0]))
plt.show()

linModel = lm.LinearRegression()
linModel.fit(X_train_n,y_train)
print(linModel.coef_)

y_test_p=linModel.predict(X_test_n)

plt.scatter(x=np.transpose(X_test[:,0]),y=y_test,c='b',s=10)
plt.scatter(x=np.transpose(X_test[:,0]),y=y_test_p,c='r',s=10)
plt.show()

MAE=sklearn.metrics.mean_absolute_error(y_test,y_test_p)
MSE=sklearn.metrics.mean_squared_error(y_test,y_test_p)
RMSE=math.sqrt(MSE)
MAPE=sklearn.metrics.mean_absolute_percentage_error(y_test,y_test_p)
R2=sklearn.metrics.r2_score(y_test,y_test_p)

print(MAE)
print(MSE)
print(RMSE)
print(MAPE)
print(R2)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.25,random_state=1)

ss = StandardScaler()
X_train_n=ss.fit_transform(X_train)
X_test_n=ss.transform(X_test)

linModel = lm.LinearRegression()
linModel.fit(X_train_n,y_train)
print(linModel.coef_)

y_test_p=linModel.predict(X_test_n)

MAE=sklearn.metrics.mean_absolute_error(y_test,y_test_p)
MSE=sklearn.metrics.mean_squared_error(y_test,y_test_p)
RMSE=math.sqrt(MSE)
MAPE=sklearn.metrics.mean_absolute_percentage_error(y_test,y_test_p)
R2=sklearn.metrics.r2_score(y_test,y_test_p)

print(MAE)
print(MSE)
print(RMSE)
print(MAPE)
print(R2)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3,random_state=1)

ss = StandardScaler()
X_train_n=ss.fit_transform(X_train)
X_test_n=ss.transform(X_test)

linModel = lm.LinearRegression()
linModel.fit(X_train_n,y_train)
print(linModel.coef_)

y_test_p=linModel.predict(X_test_n)

MAE=sklearn.metrics.mean_absolute_error(y_test,y_test_p)
MSE=sklearn.metrics.mean_squared_error(y_test,y_test_p)
RMSE=math.sqrt(MSE)
MAPE=sklearn.metrics.mean_absolute_percentage_error(y_test,y_test_p)
R2=sklearn.metrics.r2_score(y_test,y_test_p)

print(MAE)
print(MSE)
print(RMSE)
print(MAPE)
print(R2)