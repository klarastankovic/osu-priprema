import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# a) prikaz podataka
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr',
            edgecolors='black', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr',
            marker='x', linewidths=1.5, label='Test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training and Test Data')
plt.legend()


# b) model logističke regresije
model = LogisticRegression()
model.fit(X_train, y_train)


# c) parametri modela i granica odluke
theta0 = model.intercept_[0]
theta1, theta2 = model.coef_[0]

print('Model parameters:')
print(f'theta0 (intercept) = {theta0:.4f}')
print(f'theta1 (x1) = {theta1:.4f}')
print(f'theta2 (x2) = {theta2:.4f}')

x1_range = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
x2_boundary = -(theta0 + theta1 * x1_range) / theta2
  
plt.figure()
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='red',  edgecolors='black', label='1')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='blue', edgecolors='black', label='2')
plt.plot(x1_range, x2_boundary, 'k--', label='Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training Data and Decision Boundary')
plt.legend()


# d) klasifikacija testnog skupa - metrike i matrica zabune
y_test_p = model.predict(X_test)

acc  = accuracy_score(y_test, y_test_p)
prec = precision_score(y_test, y_test_p)
rec  = recall_score(y_test, y_test_p)
f1 = f1_score(y_test, y_test_p)

print('\nTest set performance:')
print(f'Accuracy: {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall: {rec:.4f}')
print(f'F1-Score: {f1:.4f}')

cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay(cm)
disp.plot()


# e) prikaz testnog skupa
correct = y_test == y_test_p
incorrect = ~correct

plt.figure()
plt.scatter(X_test[correct, 0], X_test[correct, 1],
            c='green', edgecolors='black', linewidths=0.5, label='Correct')
plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1],
            c='black', edgecolors='black', linewidths=0.5, label='Incorrect')
plt.plot(x1_range, x2_boundary, 'k--', label='Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Test Data with Predictions')
plt.legend()
plt.show()