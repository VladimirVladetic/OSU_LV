import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt("data.csv",skiprows=1,delimiter=",")
print('Broj mjerenja ',len(data))

x=data[:,1]
y=data[:,2]

plt.scatter(x,y,s=1)
plt.show()

xx=data[0:len(data):50,1]
yy=data[0:len(data):50,2]

plt.scatter(xx,yy,s=1)
plt.show()

print('Max ',max(x))
print('Min ',min(x))
print('Srednja vrijednost ',np.mean(x))

ind=(data[:,0]==1)

a=[]
b=[]

for i in range(0,len(x)):
    if ind[i]==True:
        a.append(x[i])
    else:
        b.append(x[i])

print('Max muški ',max(a))
print('Min muški ',min(a))
print('Srednja vrijednost muški ',np.mean(a))

print('Max ženski ',max(b))
print('Min ženski ',min(b))
print('Srednja vrijednost ženski ',np.mean(b))

