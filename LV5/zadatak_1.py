import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.colors as clt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, accuracy_score, recall_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a) x1-x2 ravnina
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,label="Train",cmap='viridis')
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,marker='x',label="Test",cmap='viridis')
plt.title("x1-x2 Ravnina")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

#b) Izrada modela logističke regresije
logisticRegModel=LogisticRegression()
logisticRegModel.fit(X_train,y_train)

#c) Parametri modela
print('Nulti parametar:',logisticRegModel.intercept_)
print('Ostali parametri:',logisticRegModel.coef_)

plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap='viridis')
plt.title("x1-x2 Ravnina, Train podaci")
plt.xlabel("x1")
plt.ylabel("x2")

# Crtanje granice odluke
a = -logisticRegModel.coef_[0][0] / logisticRegModel.coef_[0][1]
b = -logisticRegModel.intercept_ / logisticRegModel.coef_[0][1]
x1 = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 100)
x2 = a*x1 + b

plt.plot(x1, x2, label='Granica odluke')
plt.legend()
plt.show()

#d)
y_test_p=logisticRegModel.predict(X_test)

# Izrada i prikaz matrice zabune
cm=confusion_matrix(y_test,y_test_p)
cm_disp=ConfusionMatrixDisplay(cm)
cm_disp.plot()
plt.show()

print('Preciznost:',precision_score(y_test,y_test_p))
print('Točnost:',accuracy_score(y_test,y_test_p))
print('Odziv:',recall_score(y_test,y_test_p))

#e)
testingArray=(y_test==y_test_p)
plt.scatter(X_test[:,0],X_test[:,1],c=testingArray,cmap=clt.ListedColormap(['black','green']))
plt.show()

