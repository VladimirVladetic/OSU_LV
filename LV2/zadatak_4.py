import numpy as np 
import matplotlib.pyplot as plt 

b=np.ones((50,50))
w=np.zeros((50,50))

stack1=np.vstack((b,w))
stack2=np.vstack((w,b))
img=np.hstack((stack2,stack1))
plt.imshow(img, cmap="gray")
plt.show()




