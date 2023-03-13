import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("road.jpg")
img=img[:,:,0].copy()

plt.imshow(img,cmap="gray",alpha=0.5)
plt.show()

img_cropp=img[0:,160:320]
plt.imshow(img_cropp,cmap="gray")
plt.show()

plt.imshow(np.rot90(img,3),cmap="gray")
plt.show()

plt.imshow(np.fliplr(img),cmap="gray")
plt.show()



