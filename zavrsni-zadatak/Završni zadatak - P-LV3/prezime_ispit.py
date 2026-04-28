# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import datetime

# 1. zadatak
# Učitavanje podataka [cite: 3, 5, 17]
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Definicija naziva klasa za vizualizaciju [cite: 3, 18]
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# #1. zadatak (0.0.17)
# #1.a) Vizualizacija po jednog primjerka iz svake klase [cite: 18]
plt.figure(figsize=(10, 5))
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# #1.b) Distribucija uzoraka (prvih 10 000 i cijeli skup) [cite: 19, 20]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Prvih 10 000 uzoraka
y_train_10k = y_train[:10000]
unique_10k, counts_10k = np.unique(y_train_10k, return_counts=True)
ax[0].bar(unique_10k, counts_10k)
ax[0].set_title("Distribucija - Prvih 10 000 uzoraka")
ax[0].set_xlabel("Klasa")
ax[0].set_ylabel("Broj uzoraka")
ax[0].set_xticks(range(10))

# Cijeli skup
unique_all, counts_all = np.unique(y_train, return_counts=True)
ax[1].bar(unique_all, counts_all)
ax[1].set_title("Distribucija - Cijeli skup podataka")
ax[1].set_xlabel("Klasa")
ax[1].set_ylabel("Broj uzoraka")
ax[1].set_xticks(range(10))

plt.show()

# 2. zadatak
# #2.a) Priprema podataka za SVM [cite: 23, 24]
# Koristimo prvih 10 000 uzoraka, preoblikujemo u 1D i normaliziramo
X_train_svm = X_train[:10000].reshape(-1, 784) / 255.0
y_train_svm = y_train[:10000]
X_test_svm = X_test.reshape(-1, 784) / 255.0

# #2.b) Izgradnja i učenje SVM modela (RBF, C=1, gamma='scale') [cite: 25, 26]
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train_svm, y_train_svm)

# #2.c) Predikcija i vrednovanje [cite: 27, 28, 52]
y_pred_svm = svm_model.predict(X_test_svm)

# Matrica zabune s upisanim vrijednostima
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=class_names)
disp_svm.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("SVM Matrica zabune")
plt.xticks(rotation=45)
plt.show()

print("SVM Izvještaj o klasifikaciji:\n", classification_report(y_test, y_pred_svm, target_names=class_names))

# #2.d) Optimalni parametri (GridSearch - primjer) [cite: 29]
# Napomena: Ovo može trajati dugo, pa je naveden manji grid
param_grid = {'C': [1, 10], 'gamma': ['scale', 0.01]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, refit=True, verbose=0)
grid.fit(X_train_svm, y_train_svm)
print(f"Najbolji SVM parametri: {grid.best_params_}")

# 3. zadatak
# #3.a) Izgradnja CNN mreže [cite: 30, 31, 33, 34, 35, 36, 37, 38]
# Priprema podataka (28x28x1) i normalizacija
X_train_cnn = X_train[:10000].reshape(-1, 28, 28, 1) / 255.0
X_test_cnn = X_test.reshape(-1, 28, 28, 1) / 255.0

# Enkodiranje oznaka u kategoričke [cite: 32, 42]
y_train_cnn = keras.utils.to_categorical(y_train[:10000], 10)
y_test_cnn = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Ispis informacija o mreži [cite: 39]
model.summary()

# #3.b) Podešavanje procesa treniranja [cite: 41, 42, 43, 44]
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# #3.c) Tensorboard callback [cite: 45]
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [
    #tensorboard_callback
             ]

# #3.d) Treniranje mreže [cite: 46, 47]
model.fit(X_train_cnn, y_train_cnn, batch_size=64, epochs=5, validation_split=0.1, callbacks=callbacks)

# #3.e) Promjena veličine batch-a (npr. 512) [cite: 48]
model.fit(X_train_cnn, y_train_cnn, batch_size=512, epochs=2, validation_split=0.1, callbacks=callbacks)

# #3.g) Pohrana modela na disk [cite: 50]
model.save("model.keras")

# #3.h) Predikcija i matrica zabune za CNN [cite: 51, 52]
# Učitavanje modela (demonstracija)
loaded_model = keras.models.load_model("model.keras")
y_pred_cnn = np.argmax(loaded_model.predict(X_test_cnn), axis=1)
y_true_cnn = np.argmax(y_test_cnn, axis=1)

cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title("CNN Matrica zabune")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.show()