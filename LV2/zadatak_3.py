import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("LV2\\road.jpg")
img=img[:,:,0].copy()

#a) Posvijetli sliku
bright_img=np.clip(img.astype(np.int64)+150,0,255)

plt.imshow(img,cmap="gray")
plt.title("Normalna slika")
plt.show()
plt.imshow(bright_img,cmap="gray")
plt.title("Svijetlija slika")
plt.show()


#b)
# Prikaz dimenzija slike
print(img.shape)

# Prikaz druge četvrtine slike po širini
plt.imshow(img[:,160:320],cmap="gray")
plt.title("Prikaz druge četvrtine slike po širini")
plt.show()

#c)
plt.imshow(np.rot90(img,3),cmap="gray")
plt.title("Rotirana slika za 90° u smjeru kazaljke na saatu")
plt.show()

#d)
plt.imshow(np.fliplr(img),cmap="gray")
plt.title("Inverz po y osi slika")
plt.show()