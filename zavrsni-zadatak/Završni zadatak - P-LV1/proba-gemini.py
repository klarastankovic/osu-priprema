import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Učitavanje dataseta
california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target
df = california.frame

# 1.a) Odgovori na pitanja o skupu podataka
broj_uzoraka = df.shape[0]
broj_znacajki = X.shape[1]
nazivi_znacajki = X.columns.tolist()
izlazna_velicina = california.target_names[0]
min_cijena = y.min()
max_cijena = y.max()

print("1.a) Informacije o datasetu:")
print(f"Broj uzoraka: {broj_uzoraka}")
print(f"Broj ulaznih veličina (značajki): {broj_znacajki}")
print(f"Nazivi značajki: {nazivi_znacajki}")
print(f"Izlazna veličina: {izlazna_velicina}")
print(f"Raspon izlazne veličine: od {min_cijena} do {max_cijena}")
print("========================================\n")

# 1.b) Podjela podataka i korelacijske matrice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Korelacijska matrica ulaznih veličina (prikaz s upisanim vrijednostima)
plt.figure(figsize=(10, 8))
korelacija_ulaza = X_train.corr()
sns.heatmap(korelacija_ulaza, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelacijska matrica ulaznih veličina")
plt.show()

# Korelacija ulaznih veličina i cijene (izlazne veličine)
korelacija_s_cijenom = X_train.corrwith(y_train)
print("1.b) Korelacija ulaznih veličina s cijenom nekretnine:")
print(korelacija_s_cijenom)
print("========================================\n")

# 1.c) Dijagram raspršenja ovisnosti cijene o ulaznim veličinama
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
fig.suptitle('Ovisnost cijene nekretnine o ulaznim veličinama', fontsize=16)

for i, col in enumerate(X.columns):
    ax = axes[i // 4, i % 4]
    ax.scatter(X_train[col], y_train, color='green', alpha=0.5, s=10)
    ax.set_title(f"Cijena vs {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Cijena")

plt.tight_layout()
plt.show()

# 1.d) Kutijasti dijagrami distribucije ulaznih veličina
plt.figure(figsize=(12, 6))
sns.boxplot(data=X_train)
plt.title("Distribucija ulaznih veličina (Boxplot)")
plt.xticks(rotation=45)
plt.yscale('log') # Skalirano logaritamski zbog velikih razlika u rasponima (npr. Population vs MedInc)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Skaliranje (normalizacija) podataka na temelju skupa za učenje
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2.a) Izgradnja modela linearne regresije
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train_scaled, y_train)

# 2.b) Predikcija i vrednovanje modela
y_pred_lin = lin_reg_model.predict(X_test_scaled)

rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
mae_lin = mean_absolute_error(y_test, y_pred_lin)
mape_lin = mean_absolute_percentage_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print("2.b) Metrike modela linearne regresije:")
print(f"RMSE: {rmse_lin:.4f}")
print(f"MAE: {mae_lin:.4f}")
print(f"MAPE: {mape_lin:.4f}")
print(f"R2: {r2_lin:.4f}")
print("========================================\n")

# 2.c) Vizualizacija stvarnih i predviđenih vrijednosti
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lin, color='orange', alpha=0.5, label='Predviđeno vs Stvarno')

# Pravac idealne ovisnosti
min_val = min(y_test.min(), y_pred_lin.min())
max_val = max(y_test.max(), y_pred_lin.max())
plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=2, label='Idealna ovisnost')

plt.title("Stvarne vs Predviđene cijene (Linearna regresija)")
plt.xlabel("Stvarne cijene")
plt.ylabel("Predviđene cijene")
plt.legend()
plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import os

# 3.a) Izgradnja neuronske mreže
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(50, activation='relu'),
    Dense(1)
])

print("3.a) Informacije o mreži (Summary):")
model_nn.summary()
print("========================================\n")

# 3.b) Podešavanje procesa treniranja
model_nn.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 3.c) Omogućavanje praćenja učenja pomoću Tensorboarda
log_dir = os.path.join("logs", "fit", "run_normal_batch")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 3.d) Pokretanje učenja (batch=32, epochs=50, validation=10%)
print("3.d) Pokretanje učenja (batch_size=32)...")
history_normal = model_nn.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[tensorboard_callback],
    verbose=0 # Postavljeno na 0 da ne ispisuje svaku epohu u terminal, promijeniti u 1 za prikaz
)

# 3.e) Drastična promjena batch size-a
# Izgradnja i kompajliranje nove identične mreže za testiranje velikog batcha
model_nn_large_batch = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(50, activation='relu'),
    Dense(1)
])
model_nn_large_batch.compile(loss='mse', optimizer='adam', metrics=['mae'])

log_dir_large = os.path.join("logs", "fit", "run_large_batch")
tensorboard_callback_large = TensorBoard(log_dir=log_dir_large, histogram_freq=1)

print("3.e) Pokretanje učenja s drastično većim batch size-om (batch_size=1024)...")
history_large = model_nn_large_batch.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=1024,
    validation_split=0.1,
    callbacks=[tensorboard_callback_large],
    verbose=0
)

# 3.f) Promatranje u Tensorboardu
# Upute za pokretanje u terminalu: tensorboard --logdir logs/fit

# 3.g) Pohrana modela na tvrdi disk (pohranjujemo onaj s batch=32)
model_path = 'model_california.h5'
model_nn.save(model_path)
print(f"3.g) Model je pohranjen na lokaciju: {model_path}")

# Učitavanje modela
ucitani_model = load_model(model_path)

# 3.h) Predikcija s učitanim modelom i vrednovanje
y_pred_nn = ucitani_model.predict(X_test_scaled).flatten()

rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mape_nn = mean_absolute_percentage_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("3.h) Metrike modela neuronske mreže:")
print(f"RMSE: {rmse_nn:.4f}")
print(f"MAE: {mae_nn:.4f}")
print(f"MAPE: {mape_nn:.4f}")
print(f"R2: {r2_nn:.4f}")