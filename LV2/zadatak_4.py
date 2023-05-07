import numpy as np
import matplotlib.pyplot as plt

white=np.ones((50,50))
black=np.zeros((50,50))
stack1=np.hstack((black,white))
stack2=np.hstack((white,black))
img=np.vstack((stack1,stack2))
plt.imshow(img,cmap="gray")
plt.show()