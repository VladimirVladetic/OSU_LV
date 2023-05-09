import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("LV6\Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

#############################
# Zad 1

#a) Napravi KNN
KNN_model=KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n,y_train)

y_test_p_KNN=KNN_model.predict(X_test_n)
y_train_p_KNN=KNN_model.predict(X_train_n)

# Računanje točnosti kod podataka i za učenje i testiranje
print('KNN Model, N=5')
print('Točnost kod podataka za učenje:',accuracy_score(y_train,y_train_p_KNN))
print('Točnost kod podataka za testiranje:',accuracy_score(y_test,y_test_p_KNN))

plot_decision_regions(X_train_n,y_train,classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("K=5, Točnost kod podataka za učenje: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

# Granica odluke nije više pravac

# Zad 2), ostali KNN modeli

KNN_model_N1=KNeighborsClassifier(n_neighbors=1)
KNN_model_N1.fit(X_train_n,y_train)

y_test_p_KNN_N1=KNN_model_N1.predict(X_test_n)
y_train_p_KNN_N1=KNN_model_N1.predict(X_train_n)

print('KNN Model, N=1')
print('Točnost kod podataka za učenje:',accuracy_score(y_train,y_train_p_KNN_N1))
print('Točnost kod podataka za testiranje:',accuracy_score(y_test,y_test_p_KNN_N1))

plot_decision_regions(X_train_n,y_train,classifier=KNN_model_N1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("K=1, Točnost kod podataka za učenje: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN_N1))))
plt.tight_layout()
plt.show()

KNN_model_N100=KNeighborsClassifier(n_neighbors=100)
KNN_model_N100.fit(X_train_n,y_train)

y_test_p_KNN_N100=KNN_model_N100.predict(X_test_n)
y_train_p_KNN_N100=KNN_model_N100.predict(X_train_n)

print('KNN Model, N=100')
print('Točnost kod podataka za učenje:',accuracy_score(y_train,y_train_p_KNN_N100))
print('Točnost kod podataka za testiranje:',accuracy_score(y_test,y_test_p_KNN_N100))

plot_decision_regions(X_train_n,y_train,classifier=KNN_model_N100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("K=1, Točnost kod podataka za učenje: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN_N100))))
plt.tight_layout()
plt.show()

# 1-Overfitting 100-Underfitting

# Zadatak 2, najbolji KNN, N=7

KNN_model2=KNeighborsClassifier()
param_grid_KNN={'n_neighbors': np.arange(1,100)}
knn_gscv=GridSearchCV(KNN_model2,param_grid_KNN,cv=5,scoring='accuracy',n_jobs=-1)
knn_gscv.fit(X_train_n, y_train )
print('Najbolja vrijednost parametra K: ',knn_gscv.best_params_)
#print(knn_gscv.best_score_)

# Zadatak 3, SVM

# Klasicni SVM RBF

SVM_model=svm.SVC(kernel='rbf',gamma=1,C=0.1)
SVM_model.fit(X_train_n,y_train)

y_test_p_SVM=SVM_model.predict(X_test_n)
y_train_p_SVM=SVM_model.predict(X_train_n)

print('SVM RBF, G=1, C=0.1')
print('Točnost kod podataka za učenje:',accuracy_score(y_train,y_train_p_SVM))
print('Točnost kod podataka za testiranje:',accuracy_score(y_test,y_test_p_SVM))

plot_decision_regions(X_train_n,y_train,classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("RBF, g=1, C=0.1, Točnost kod podataka za učenje: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()

# Sigmoid SVM

SVM_model_sigmoid=svm.SVC(kernel='sigmoid',gamma=1,C=0.1)
SVM_model_sigmoid.fit(X_train_n,y_train)

y_test_p_SVM_sigmoid=SVM_model_sigmoid.predict(X_test_n)
y_train_p_SVM_sigmoid=SVM_model_sigmoid.predict(X_train_n)

print('SVM Sigmoid, G=1, C=0.1')
print('Točnost kod podataka za učenje:',accuracy_score(y_train,y_train_p_SVM_sigmoid))
print('Točnost kod podataka za testiranje:',accuracy_score(y_test,y_test_p_SVM_sigmoid))

plot_decision_regions(X_train_n,y_train,classifier=SVM_model_sigmoid)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Sigmoid, g=1, C=0.1, Točnost kod podataka za učenje: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM_sigmoid))))
plt.tight_layout()
plt.show()

# Alternativni SVM RBF

SVM_model_alt=svm.SVC(kernel='rbf',gamma=10,C=1)
SVM_model_alt.fit(X_train_n,y_train)

y_test_p_SVM_alt=SVM_model_alt.predict(X_test_n)
y_train_p_SVM_alt=SVM_model_alt.predict(X_train_n)

print('SVM RBF, G=10, C=1')
print('Točnost kod podataka za učenje:',accuracy_score(y_train,y_train_p_SVM_alt))
print('Točnost kod podataka za testiranje:',accuracy_score(y_test,y_test_p_SVM_alt))

plot_decision_regions(X_train_n,y_train,classifier=SVM_model_alt)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("RBF, g=10, C=1, Točnost kod podataka za učenje: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM_alt))))
plt.tight_layout()
plt.show()

# Sigmoid je dao najgore rezultate, povecanjem gamma i C kod rbf kernela je povecalo točnost kod podataka za učenje i za testiranje

# Zadatak 4

# Najbolji parametri za SVM , C=1, gammma=1

SVM_model2=svm.SVC(kernel='rbf',gamma=1,C=0.1)
SVM_model2.fit(X_train_n,y_train)
param_grid_SVM={'C':[0.01,0.1,1,5,10,20,100,200],'gamma':[10,1,0.1,0.01,0.0001]}
svm_gscv = GridSearchCV (SVM_model2,param_grid_SVM,cv=5,scoring='accuracy',n_jobs=-1)
svm_gscv.fit(X_train_n,y_train)
print('Optimalne vrijednosti hiperparametara su:',svm_gscv.best_params_)
#print(svm_gscv.best_score_)