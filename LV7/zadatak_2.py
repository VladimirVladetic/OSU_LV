import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
def img_quant_with_elbow(string):
    img = Image.imread(string)

    # prikazi originalnu sliku
    #plt.figure()
    #plt.title("Originalna slika")
    #plt.imshow(img)
    #plt.tight_layout()
    #plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    unique_rows=np.unique(img_array, axis=0)

    #broj razlicitih boja
    print('Broj boja u slici:',len(unique_rows))
    
    #elbow algoritam, pronalaženje najboljeg broja clustera
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(img_array)
        distortions.append(kmeanModel.inertia_)

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
    #učenje modela
    km=KMeans(n_clusters=3,init='k-means++',n_init=5,random_state =0 )
    km.fit(img_array)
    labels=km.predict(img_array)
    
    #zamjena boja sa vrijednosti centra
    cluster_centers=km.cluster_centers_
    for i in range(len(img_array)):
        img_array_aprox[i]=cluster_centers[labels[i]]

    img_quant=np.reshape(img_array_aprox,(w,h,d))
    img_quant=(img_quant*255).astype(np.uint8)

    plt.title("Kvantizirana slika")
    plt.imshow(img_quant)
    plt.tight_layout()
    plt.show()
    
    
    #binarne slike
    km=KMeans(n_clusters=5,init='k-means++')
    km.fit(img_array)
    img_array_p=km.predict(img_array)

    for i in range(1, 6):
        img_array_k=np.full((w*h, d), 255)
        for j in range(len(img_array_p)):
            if img_array_p[j] == i-1:
                img_array_k[j]=km.cluster_centers_[i-1]*255
        img_array_k=np.reshape(img_array_k.astype(np.uint8), (w, h, d))
        plt.figure()
        plt.imshow(img_array_k)
    plt.show()

def img_quant_predeterminedn(string,clusters):
    img = Image.imread(string)

    # prikazi originalnu sliku
    #plt.figure()
    #plt.title("Originalna slika")
    #plt.imshow(img)
    #plt.tight_layout()
    #plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    unique_rows=np.unique(img_array, axis=0)

    #broj razlicitih boja
    #print(len(unique_rows))
    
    km=KMeans(n_clusters=clusters,init ='k-means++',n_init=5,random_state =0 )
    km.fit(img_array)
    labels=km.predict(img_array)

    cluster_centers=km.cluster_centers_
    for i in range(len(img_array)):
        img_array_aprox[i]=cluster_centers[labels[i]]

    img_quant=np.reshape(img_array_aprox,(w,h,d))
    img_quant=(img_quant*255).astype(np.uint8)

    plt.title("Kvantizirana slika")
    plt.imshow(img_quant)
    plt.tight_layout()
    plt.show()
    
    
    

# img_quant_predeterminedn("imgs\\test_1.jpg",3)
# img_quant_predeterminedn("imgs\\test_1.jpg",5)
# img_quant_predeterminedn("imgs\\test_1.jpg",10)

# img_quant_predeterminedn("imgs\\test_2.jpg",3)
# img_quant_predeterminedn("imgs\\test_2.jpg",5)
# img_quant_predeterminedn("imgs\\test_2.jpg",10)

# img_quant_predeterminedn("imgs\\test_3.jpg",3)
# img_quant_predeterminedn("imgs\\test_3.jpg",5)
# img_quant_predeterminedn("imgs\\test_3.jpg",10)

# img_quant_predeterminedn("imgs\\test_4.jpg",3)
# img_quant_predeterminedn("imgs\\test_4.jpg",5)
# img_quant_predeterminedn("imgs\\test_4.jpg",10)

# img_quant_predeterminedn("imgs\\test_5.jpg",3)
# img_quant_predeterminedn("imgs\\test_5.jpg",5)
# img_quant_predeterminedn("imgs\\test_5.jpg",10)

# img_quant_predeterminedn("imgs\\test_6.jpg",3)
# img_quant_predeterminedn("imgs\\test_6.jpg",5)
# img_quant_predeterminedn("imgs\\test_6.jpg",10)

#povecanjem broja centara slika je detaljnija

img_quant_with_elbow('LV7/imgs/test_1.jpg')


