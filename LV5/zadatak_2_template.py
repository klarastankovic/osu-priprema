import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

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
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


# a) stupčasti dijagram - broj primjera po klasi
classes, train_counts = np.unique(y_train, return_counts=True)
_, test_counts = np.unique(y_test, return_counts=True)

x_pos = np.arange(len(classes))
width = 0.35

plt.figure()
plt.bar(x_pos - width/2, train_counts, width, label='Train', color='lightblue')
plt.bar(x_pos + width/2, test_counts,  width, label='Test', color='salmon')
plt.xticks(x_pos, labels.values())
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Distribution of Samples by Species')
plt.legend()


# b) modle logističke regresije
model = LogisticRegression()
model.fit(X_train, y_train)


# c) parametri modela - OvR ili multinomijalna logistička regresija
print('\nModel parameters:')
print(f'Intercept shape: {model.intercept_.shape} ({model.intercept_})')
print(f'Coef shape: {model.coef_.shape}')
for k in range(model.coef_.shape[0]):
    print(f'{k} ({labels[k]}): theta = {model.coef_[k]}')


# d) decision regions
plot_decision_regions(X=X_train, y=y_train.ravel(), classifier=model)
plt.xlabel('Bill Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.title('Decision Regions for Training Data')
plt.legend(loc='upper left')


# e) klasifikacija testnog skupa - metrike i matrica zabune
y_test_p = model.predict(X_test)

cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

print("\nClassification report:")
print(classification_report(y_test, y_test_p,
                             target_names=[labels[i] for i in range(3)]))


# f) dodavanje još ulaznih veličina
input_variables_ext = ['bill_length_mm', 'bill_depth_mm',
                        'flipper_length_mm', 'body_mass_g']
 
X_ext = df[input_variables_ext].to_numpy()
X_train_ext, X_test_ext, _, _ = train_test_split(
    X_ext, y, test_size=0.2, random_state=123)
 
model_ext = LogisticRegression(max_iter=5000)
model_ext.fit(X_train_ext, y_train)
 
y_pred_ext = model_ext.predict(X_test_ext)

print("Classification report (extended model):")
print(classification_report(y_test, y_pred_ext,
                             target_names=[labels[i] for i in range(3)]))

plt.show()