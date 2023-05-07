import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('LV2\data.csv',skiprows=1,delimiter=',')

#a)
print('Izvršeno je', len(data),'mjerenja')

#b)
plt.scatter(data[:,1],data[:,2],s=1)
plt.title("Odnos visine i mase osoba")
plt.show()

#c) odnos visine i mase svake 50. osobe
x=data[0:len(data):50,1]
y=data[0:len(data):50,2]

plt.scatter(x,y,s=1)
plt.title("Odnos visine i mase svake pedesete osobe")
plt.show()


print('Srednja vrijednost svih:',np.mean(data[:,1]))
print('Minimalna vrijednost svih:',min(data[:,1]))
print('Maksimalna vrijednost svih:',max(data[:,1]))

ind=(data[:,0]==1)

muski=[]
zenski=[]

for i in range(0,len(data)):
    if ind[i]==True:
        muski.append(data[i,1])
    else:
        zenski.append(data[i,1])

print('Srednja vrijednost muskih:',np.mean(muski))
print('Minimalna vrijednost muskih:',min(muski))
print('Maksimalna vrijednost muskih:',max(muski))

print('Srednja vrijednost zenskih:',np.mean(zenski))
print('Minimalna vrijednost zenskih:',min(zenski))
print('Maksimalna vrijednost zenskih:',max(zenski))

