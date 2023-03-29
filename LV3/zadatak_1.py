import pandas as pd
import math
import matplotlib.pyplot as plt


data = pd.read_csv('data_C02_emission.csv')

print('Broj vrijednosti:', len(data))

print(data.info())

print(data.isnull().sum())

data.dropna(axis=0)

data.drop_duplicates()

data[data.select_dtypes(['object']).columns] = data.select_dtypes(
    ['object']).apply(lambda x: x.astype('category'))

subdata = data[["Fuel Consumption City (L/100km)", "Make", "Model"]]

print(subdata.nsmallest(3, "Fuel Consumption City (L/100km)"))

print(subdata.nlargest(3, "Fuel Consumption City (L/100km)"))

print(len(data[(data["Engine Size (L)"] > 2.5)
      & (data["Engine Size (L)"] < 3.5)]))

enginedata = data[(data["Engine Size (L)"] > 2.5) &
                  (data["Engine Size (L)"] < 3.5)]

print(enginedata["CO2 Emissions (g/km)"].mean())

print(len(data[(data["Make"] == "Audi")]))

audidata = data[(data["Make"] == "Audi") & (data["Cylinders"] == 4)]

print(audidata["CO2 Emissions (g/km)"].mean())

print(len(data[(data["Cylinders"] % 2 == 0)]))

print(data.groupby('Cylinders')["CO2 Emissions (g/km)"].mean())

diesel = data[data['Fuel Type'] == 'D']
print(diesel['Fuel Consumption City (L/100km)'].mean())
reg_gas = data[data['Fuel Type'] == 'X']
print(reg_gas['Fuel Consumption City (L/100km)'].mean())
print(diesel['Fuel Consumption City (L/100km)'].median())
print(reg_gas['Fuel Consumption City (L/100km)'].median())

print(data[(data["Cylinders"] == 4) & (data["Fuel Type"] == "D")]
      ["Fuel Consumption City (L/100km)"].max())

print(len(data[(data["Transmission"].str.startswith("M"))]))

pd.set_option('display.max_columns', None)
print(data.corr(numeric_only=True))
