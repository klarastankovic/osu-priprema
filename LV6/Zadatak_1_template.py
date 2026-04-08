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
from sklearn.model_selection import cross_val_score

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
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
#plt.show()

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


# Zadatak 6.5.1. KNN
# 1. K = 5
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_n, y_train)

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print("\nKNN (K=5): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("K=5 Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()

# 2. K = 1
KNN_model = KNeighborsClassifier(n_neighbors = 1)
KNN_model.fit(X_train_n, y_train)

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print("\nKNN (K=1): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("K=1 Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()

# 2. K = 100
KNN_model = KNeighborsClassifier(n_neighbors = 100)
KNN_model.fit(X_train_n, y_train)

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print("\nKNN (K=100): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("K=100 Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()


# Zadatak 6.5.2. KNN cross validation
KNN_model = KNeighborsClassifier(n_neighbors = 5)

param_grid_knn = {'n_neighbors': np.arange(1, 100)}
knn_gscv = GridSearchCV(KNN_model, param_grid_knn, cv=5, scoring='accuracy')
knn_gscv.fit(X_train_n, y_train)

print("\nKNN (cross validation)")
print("Optimalni broj susjeda (K):", knn_gscv.best_params_['n_neighbors'])
print("Najbolja tocnost unakrsne validacije:", "{:0.3f}".format(knn_gscv.best_score_))

k_values = knn_gscv.cv_results_['param_n_neighbors'].data
cv_scores = knn_gscv.cv_results_['mean_test_score']

plt.figure()
plt.plot(k_values, cv_scores)
plt.xlabel('Broj susjeda K')
plt.ylabel('CV tocnost (Mean Test Score)')
plt.title('Ovisnost tocnosti o broju susjeda K')
plt.grid(True)

best_k = knn_gscv.best_params_['n_neighbors']
best_score = knn_gscv.best_score_
plt.scatter(best_k, best_score, color='red', s=5, label=f'Optimalan K={best_k}', zorder=5)
plt.legend()
plt.tight_layout()

plt.show()


# Zadatak 6.5.3. SVM
SVM_model = svm.SVC(kernel='rbf', gamma=1.0, C=1.0)
SVM_model.fit(X_train_n, y_train)

y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)

print("\nSVM (RBF kernel, C=1.0, gamma=1.0):")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_SVM)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_SVM)))

plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("SVM RBF (C=1.0, gamma=1.0)\n"
          f"Tocnost - train: {accuracy_score(y_train, y_train_p_SVM):.3f}, test: {accuracy_score(y_test, y_test_p_SVM):.3f}")
plt.legend(loc='upper left')
plt.tight_layout()

configurations = [
    {'kernel': 'linear', 'C': 1.0, 'gamma': 1.0},
    {'kernel': 'poly', 'C': 1.0, 'gamma': 1.0},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 0.1},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 10.0},
    {'kernel': 'rbf', 'C': 0.1, 'gamma': 1.0},
    {'kernel': 'rbf', 'C': 10.0, 'gamma': 1.0}
]

for config in configurations:
    svm_m = svm.SVC(kernel=config['kernel'], C=config['C'], gamma=config['gamma'])
    svm_m.fit(X_train_n, y_train)
    
    train_acc = accuracy_score(y_train, svm_m.predict(X_train_n))
    test_acc = accuracy_score(y_test, svm_m.predict(X_test_n))
    
    plot_decision_regions(X_train_n, y_train, classifier=svm_m)
    plt.title(f"SVM kernel: {config['kernel']}, C: {config['C']}, gamma: {config['gamma']}\n"
              f"Tocnost - train: {train_acc:.3f}, test: {test_acc:.3f}")
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.tight_layout()


# Zadatak 6.5.4. SVM cross validation
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}
svm_gscv = GridSearchCV(svm.SVC(), param_grid_svm, cv=5, scoring='accuracy')
svm_gscv.fit(X_train_n, y_train)

print("\nSVM (cross validation)")
print("Najbolji parametri za SVM:", svm_gscv.best_params_)
print("Najbolja CV tocnost:", "{:0.3f}".format(svm_gscv.best_score_))

best_svm = svm_gscv.best_estimator_
y_test_p_best = best_svm.predict(X_test_n)
print("Tocnost na testnom skupu s najboljim SVM modelom: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_best)))

plot_decision_regions(X_train_n, y_train, classifier=best_svm)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title(f"Najbolji SVM: C={svm_gscv.best_params_['C']}, gamma={svm_gscv.best_params_['gamma']}\n"
          f"Tocnost - train: {accuracy_score(y_train, best_svm.predict(X_train_n)):.3f}, test: {accuracy_score(y_test, y_test_p_best):.3f}")
plt.legend(loc='upper left')
plt.tight_layout()


plt.show()