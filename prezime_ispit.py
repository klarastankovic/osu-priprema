#imports
from gc import callbacks

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)
 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#1. zadatak
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

#1.a)
print("\n1.a) Informacije o skupu podataka")
print(f"Broj uzoraka: {data.data.shape[0]}")
print(f"Broj značajki: {data.data.shape[1]}")
print(f"Nazivi značajki:\n  {list(data.feature_names)}")
print(f"\nIzlazna veličina: 'target'")
print(f"Moguće vrijednosti: {list(np.unique(data.target))}")
print(f"Nazivi klasa: {list(data.target_names)}")
print(f" 0 = {data.target_names[0]} (maligni tumor)")
print(f" 1 = {data.target_names[1]} (benigni tumor)")
print(f"\nRaspodjela klasa:\n{pd.Series(data.target).value_counts().rename({0: data.target_names[0], 1: data.target_names[1]}).to_string()}")

#1.b)
print("\n1.b) Korelacijska matrica")
corr = df[data.feature_names].corr()
 
fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(corr, annot=True, 
            annot_kws={"size": 7}, fmt=".2f", cmap="coolwarm",
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Korelacijska matrica ulaznih veličina",
             fontsize=14, pad=15)
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(rotation=0, fontsize=7)
#plt.show()

#1.c)
print("\n1.c) Kutijasti dijagrami")
fig, axes = plt.subplots(5, 6, figsize=(22, 18))
axes = axes.flatten()
for i, col in enumerate(data.feature_names):
    axes[i].boxplot(df[col], patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.7))
    axes[i].set_title(col, fontsize=7)
    axes[i].set_xlabel("Značajka", fontsize=6)
    axes[i].set_ylabel("Vrijednost", fontsize=6)
    axes[i].tick_params(labelsize=6)

for j in range(len(data.feature_names), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Kutijasti dijagrami ulaznih veličina", fontsize=14, y=1.01)
plt.tight_layout()
#plt.show()

#1.d)
print("\n1.d) Normalizacija i kutijasti dijagrami")
scaler_006 = MinMaxScaler()
X_norm_006 = scaler_006.fit_transform(df[data.feature_names])
df_norm = pd.DataFrame(X_norm_006, columns=data.feature_names)
 
fig, axes = plt.subplots(5, 6, figsize=(22, 18))
axes = axes.flatten()
for i, col in enumerate(data.feature_names):
    axes[i].boxplot(df_norm[col], patch_artist=True,
                    boxprops=dict(facecolor="#55A868", alpha=0.7))
    axes[i].set_title(col, fontsize=7)
    axes[i].set_xlabel("Značajka", fontsize=6)
    axes[i].set_ylabel("Normalizirana vrijednost", fontsize=6)
    axes[i].tick_params(labelsize=6)
for j in range(len(data.feature_names), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Kutijasti dijagrami (MinMax normalizirani podaci)",
             fontsize=14, y=1.01)
plt.tight_layout()
#plt.show()

#2. zadatak
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f"\nBroj null/NaN vrijednosti: {X.isnull().sum().sum()}")
X.dropna(inplace=True)
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

print(f"Trening skup: {X_train.shape[0]} uzoraka")
print(f"Test skup: {X_test.shape[0]} uzoraka")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

#2.a)
print("\n2.a) Treniranje SVM modela")
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svm_model.fit(X_train_s, y_train)
print("SVM model treniran.")
 
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)
 
svm_2d = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_2d.fit(X_train_pca, y_train)
 
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
 
fig, ax = plt.subplots(figsize=(10, 7))
ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
ax.contour(xx, yy, Z, colors="k", linewidths=0.8)
colors = {0: "#D62728", 1: "#1F77B4"}
labels = {0: data.target_names[0], 1: data.target_names[1]}
for cls in [0, 1]:
    mask = y_train.values == cls
    ax.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
               c=colors[cls], label=labels[cls], s=25, alpha=0.7, edgecolors="k", linewidths=0.3)
ax.set_xlabel("PCA komponenta 1", fontsize=11)
ax.set_ylabel("PCA komponenta 2", fontsize=11)
ax.set_title("SVM – Granica odluke (PCA 2D projekcija, trening skup)", fontsize=13)
ax.legend(fontsize=10)
#plt.show()

#2.b)
print("\n2.b) Klasifikacija testnog skupa")
y_train_pred = svm_model.predict(X_train_s)
y_test_pred  = svm_model.predict(X_test_s)
print(f"Klasifikacija testnog skupa provedena. Broj predikcija: {len(y_test_pred)}")

#2.c)
print("\nc) Klasifikacijske metrike")
for split, y_true, y_pred in [("TRENING", y_train, y_train_pred),
                               ("TEST",    y_test,  y_test_pred)]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    print(f"\n  [{split} SKUP]")
    print(f"Točnost (Accuracy): {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Preciznost (Precision): {prec:.4f}")
    print(f"Odziv (Recall): {rec:.4f}")

#2.d)
print("\n2.d) Matrica zabune (testni skup)")
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=data.target_names)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("Matrica zabune – SVM (testni skup)", fontsize=13)
print(f"Matrica zabune:\n{cm}")
#plt.show()

#3. zadatak
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
 
X.dropna(inplace=True)
y = y.loc[X.index]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
 
print(f"\nTrening skup: {X_train.shape[0]} uzoraka")
print(f"Test skup: {X_test.shape[0]} uzoraka")
 
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
 
n_features = X_train_s.shape[1]

#3.a)
print("\n3.a) Izgradnja neuronske mreže")
model = keras.Sequential([
    layers.Input(shape=(n_features,),  name="ulazni_sloj"),
    layers.Dense(16, activation="relu", name="skriveni_sloj_1"),
    layers.Dense(8,  activation="relu", name="skriveni_sloj_2"),
    layers.Dense(1,  activation="sigmoid", name="izlazni_sloj"),
], name="BreastCancer_NN")
 
model.summary()

#3.b)
print("\n3.b) Kompajliranje mreže")
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
print("Model kompajliran: loss=binary_crossentropy, optimizer=adam, metrics=accuracy")

#3.c)
print("\n3.c) TensorBoard callback")
my_callbacks = [
    #keras.callbacks.TensorBoard(log_dir="./logs/batch16", update_freq=100)
]

#3.d)
print("\n3.d) Treniranje: 50 epoha, batch_size=16, val_split=0.10")
model.fit(X_train_s,
            y_train,
            epochs = 50,
            batch_size = 16,
            callbacks = my_callbacks,
            validation_split = 0.10)

score = model.evaluate(X_test_s, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

#3.e)
print("\n3.e) Treniranje: 50 epoha, batch_size=128 (drastično veći batch)")
callbacks = [
    #keras.callbacks.TensorBoard(log_dir='./logs/batch128', update_freq=100)
]

model.fit(X_train_s,
            y_train,
            epochs = 50,
            batch_size = 128,
            callbacks = callbacks,
            validation_split = 0.10)

score = model.evaluate(X_test_s, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

#3.g)
model.save("breast_cancer_model.keras")
loaded_model = keras.models.load_model("breast_cancer_model.keras")

#3.h)
print("\n3.h) Predikcija i metrike (testni skup, učitani model)")
y_pred_prob = loaded_model.predict(X_test_s, verbose=0)
y_pred  = (y_pred_prob >= 0.5).astype(int).flatten()
 
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
 
print(f"Točnost (Accuracy): {acc:.4f}  ({acc*100:.2f}%)")
print(f"Preciznost (Precision): {prec:.4f}")
print(f"Odziv (Recall): {rec:.4f}")
 
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=data.target_names)
disp.plot(ax=ax, colorbar=True, cmap="Greens")
ax.set_title("Matrica zabune – Neuronska mreža (testni skup)", fontsize=13)
print(f"\nMatrica zabune (NN):\n{cm}")
plt.show()