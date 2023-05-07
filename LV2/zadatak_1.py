import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,3,3,2,1],float)
y=np.array([1,1,2,2,1],float)

plt.plot(x,y,'b', linewidth=1,marker=".",markersize=10)
plt.axis([0,4,0,4])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Naslov")
plt.show()