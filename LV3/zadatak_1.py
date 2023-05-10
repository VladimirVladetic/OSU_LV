import pandas as pd
import numpy as np

data = pd.read_csv('LV3\data_C02_emission.csv')
pd.set_option('display.max_columns', None)

#a)
print("Broj mjerenja u DataFrame:",len(data))

print("Tipovi podataka za veličine:",data.dtypes)

# Brisanje svih redova di nedostaje vrijednost
#data.dropna(axis=0)
# Brisanje duplikatskih redova
#data.drop_duplicates()

#p Provjera koliko ima izostalih vrijednosti
print(data.isnull().sum())
print("Nema izostalih vrijednosti")

duplicates = data.duplicated()
print(data[duplicates])
print("Nema duplikata")

# Konverzija iz object u category
data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))

#b)
data_subset_fuel=data[['Make','Model','Fuel Consumption City (L/100km)',]]

print('3 auta koji najmanje troše:',data_subset_fuel.nsmallest(3,'Fuel Consumption City (L/100km)'))
print('3 auta koji najviše troše:',data_subset_fuel.nlargest(3,'Fuel Consumption City (L/100km)'))

#c)
data_subset_motor_size=data[(data['Engine Size (L)']>=2.5) & (data['Engine Size (L)']<=3.5)]

print('Broj vozila sa motorima veličine između 2.5 i 3.5:',len(data_subset_motor_size))
print('Prosječna ispuštanje CO2 emisija ovih vozila:',np.mean(data_subset_motor_size['CO2 Emissions (g/km)']))

#d)

data_subset_audi=data[(data['Make']=='Audi')]

print('Broj Audi vozila:',len(data_subset_audi))

data_subset_audi_cylinders=data_subset_audi[(data_subset_audi['Cylinders']==4)]

print('CO2 emisije Audi auta sa 4 cilindra:',np.mean(data_subset_audi_cylinders['CO2 Emissions (g/km)']))

#e)
print('Broj vozila s parnim brojem cilindara:',len(data[(data['Cylinders']%2==0)]))

print('CO2 potrošnja s obzirom na broj cilindara:',data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean())

#f)
data_subset_diesel=data[(data['Fuel Type']=='D')]
data_subset_gas=data[(data['Fuel Type']=='X')]

print('Prosječna potrošnja dizelaša u gradu:',np.mean(data_subset_diesel['Fuel Consumption City (L/100km)']))
print('Prosječna potrošnja benzinaca u gradu:',np.mean(data_subset_gas['Fuel Consumption City (L/100km)']))

print('Median dizelaša',np.median(data_subset_diesel['Fuel Consumption City (L/100km)']))
print('Median benzinaca',np.median(data_subset_gas['Fuel Consumption City (L/100km)']))

#g)
data_subset_diesel_4=data[(data['Fuel Type']=='D') & (data['Cylinders']==4)]

print('Dizelaš sa 4 cilindra s najvećom gradskom potrošnjom:')
print(data_subset_diesel_4.nlargest(1,'Fuel Consumption City (L/100km)')[['Make','Model','Fuel Consumption City (L/100km)']])

#h)
data_subset_manual=data[(data['Transmission'].str.startswith('M'))]

print('Broj auta sa mjenjačem:',len(data_subset_manual))

#i)
print(data.corr(numeric_only=True))

