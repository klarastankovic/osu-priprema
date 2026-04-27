# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

import os
import datetime

# 1. zadatak
housing = fetch_california_housing(as_frame=True)
df = housing.frame
 
X_all = housing.data
y_all = housing.target
 
# 1.a) Informacije o datasetu
print("\n--- 1.a) Informacije o datasetu ---")
print(f"Broj uzoraka: {X_all.shape[0]}")
print(f"Broj znacajki (ulaznih velicina): {X_all.shape[1]}")
print(f"Nazivi znacajki: {list(housing.feature_names)}")
print(f"Izlazna velicina: MedHouseVal (medijalna vrijednost kuce u 100.000 USD)")
print(f"Raspon izlazne velicine: min = {y_all.min():.4f}, max = {y_all.max():.4f}")
 
# 1.b) Train/test split, korelacijska matrica
print("\n--- 1.b) Train/test split i korelacijska matrica ---")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)
print(f"Velicina trening skupa: {X_train_raw.shape[0]}")
print(f"Velicina test skupa: {X_test_raw.shape[0]}")
 
corr_matrix = X_train_raw.corr()
print("\nKorelacijska matrica ulaznih velicina:")
print(corr_matrix.to_string())
 
corr_with_price = pd.concat([X_train_raw, y_train], axis=1).corr()["MedHouseVal"].drop("MedHouseVal")
print("\nKorelacija ulaznih velicina s cijenom nekretnine:")
print(corr_with_price.to_string())
 
# Vizualizacija korelacijske matrice
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(housing.feature_names)))
ax.set_yticks(range(len(housing.feature_names)))
ax.set_xticklabels(housing.feature_names, rotation=45, ha="right")
ax.set_yticklabels(housing.feature_names)
ax.set_title("Korelacijska matrica ulaznih velicina")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
korelacija_ulaza = X_train_raw.corr()
sns.heatmap(korelacija_ulaza, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelacijska matrica ulaznih veličina")
plt.show()
 
# 1.c) Dijagrami rasprsenja - ovisnost cijene o svakim ulaznim velicinama
print("\n--- 1.c) Dijagrami rasprsenja ---")
n_features = X_train_raw.shape[1]
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 4, figure=fig)
 
for i, feature in enumerate(housing.feature_names):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    ax.scatter(X_train_raw[feature], y_train, color="green", alpha=0.3, s=5)
    ax.set_xlabel(feature)
    ax.set_ylabel("Cijena nekretnine (MedHouseVal)")
    ax.set_title(f"Cijena vs {feature}")
 
fig.suptitle("Ovisnost cijene nekretnine o ulaznim velicinama", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
 
# 1.d) Kutijasti dijagrami distribucije ulaznih velicina
print("\n--- 1.d) Kutijasti dijagrami ---")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
 
for i, feature in enumerate(housing.feature_names):
    axes[i].boxplot(X_train_raw[feature].dropna(), patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    axes[i].set_title(f"Distribucija: {feature}")
    axes[i].set_ylabel("Vrijednost")
 
plt.suptitle("Kutijasti dijagrami distribucije ulaznih velicina", fontsize=14)
plt.tight_layout()
plt.show()

# 2. zadatak
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
 
# 2.a) Izgradnja modela linearne regresije
print("\n--- 2.a) Izgradnja modela linearne regresije ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("Model linearne regresije je uspjesno izgraden.")
print(f"Koeficijenti: {lr_model.coef_}")
print(f"Slobodni clan (intercept): {lr_model.intercept_:.4f}")
 
# 2.b) Predikcija i vrednovanje
print("\n--- 2.b) Predikcija i vrednovanje ---")
y_pred_lr = lr_model.predict(X_test_scaled)
 
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100
r2_lr = r2_score(y_test, y_pred_lr)
 
print(f"RMSE: {rmse_lr:.4f}")
print(f"MAE: {mae_lr:.4f}")
print(f"MAPE: {mape_lr:.4f} %")
print(f"R2: {r2_lr:.4f}")
 
# 2.c) Vizualizacija stvarnih vs predvidenih vrijednosti
print("\n--- 2.c) Vizualizacija stvarnih vs predvidenih vrijednosti ---")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred_lr, color="orange", alpha=0.4, s=10,
           label="Predvidene vs stvarne vrijednosti")
 
min_val = min(y_test.min(), y_pred_lr.min())
max_val = max(y_test.max(), y_pred_lr.max())
ax.plot([min_val, max_val], [min_val, max_val], color="gray",
        linewidth=2, label="Idealna ovisnost (y = x)")
 
ax.set_xlabel("Stvarne vrijednosti cijene nekretnine")
ax.set_ylabel("Predvidene vrijednosti cijene nekretnine")
ax.set_title("Stvarne vs predvidene vrijednosti - Linearna regresija")
ax.legend()
plt.tight_layout()
plt.show()

# 3. zadatak
print("\n--- 3.a) Izgradnja neuronske mreze ---")
nn_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu", name="skriveni_sloj_1"),
    layers.Dense(50, activation="relu", name="skriveni_sloj_2"),
    layers.Dense(1, name="izlazni_sloj")
], name="neuronska_mreza_california")
 
nn_model.summary()
 
# 3.b) Konfiguracija procesa treniranja
print("\n--- 3.b) Konfiguracija treniranja ---")
nn_model.compile(
    loss="mse",
    optimizer="adam",
    metrics=["mae"]
)
print("Model kompajliran: loss=mse, optimizer=adam, metrics=mae")
 
# 3.c) Tensorboard callback
print("\n--- 3.c) TensorBoard callback ---")
log_dir_1 = os.path.join("logs", "fit", "batch32_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb_1 = TensorBoard(log_dir=log_dir_1, histogram_freq=1)
print(f"TensorBoard logovi se zapisuju u: {log_dir_1}")
print("Pokrenite 'tensorboard --logdir logs/fit' za prikaz.")
callbacks = [
    #tensorboard_cb_1
]
 
# 3.d) Ucenje mreze - 50 epoha, batch_size=32, 10% validacija
print("\n--- 3.d) Ucenje mreze (50 epoha, batch_size=32, 10% validacija) ---")
history_1 = nn_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)
 
# 3.e) Drasticna promjena velicine batch-a i ponavljanje ucenja
print("\n--- 3.e) Drasticna promjena velicine batch-a (batch_size=512) ---")
 
nn_model_2 = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu", name="skriveni_sloj_1"),
    layers.Dense(50, activation="relu", name="skriveni_sloj_2"),
    layers.Dense(1, name="izlazni_sloj")
], name="neuronska_mreza_california_v2")
 
nn_model_2.compile(
    loss="mse",
    optimizer="adam",
    metrics=["mae"]
)
 
log_dir_2 = os.path.join("logs", "fit", "batch512_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb_2 = TensorBoard(log_dir=log_dir_2, histogram_freq=1)
print(f"TensorBoard logovi se zapisuju u: {log_dir_2}")
callbacks_2 = [
    #tensorboard_cb_2
    ]
 
history_2 = nn_model_2.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=512,
    validation_split=0.1,
    callbacks=callbacks_2,
    verbose=1
)
 
# 3.f) Promatranje parametara u Tensorboardu
print("\n--- 3.f) Promatranje parametara u TensorBoardu ---")
print("Pokrenite 'tensorboard --logdir logs/fit' za vizualizaciju i usporedbu.")
 
# 3.g) Pohrana modela na tvrdi disk i ucitavanje
print("\n--- 3.g) Pohrana i ucitavanje modela ---")
model_path = "nn_model_california.keras"
nn_model.save(model_path)
print(f"Model pohranjen na: {model_path}")
 
loaded_model = keras.models.load_model(model_path)
print("Model uspjesno ucitan s diska.")
loaded_model.summary()
 
# 3.h) Predikcija i vrednovanje ucitanog modela
print("\n--- 3.h) Predikcija i vrednovanje neuronske mreze ---")
y_pred_nn = loaded_model.predict(X_test_scaled, batch_size=32).flatten()
 
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mape_nn = np.mean(np.abs((y_test - y_pred_nn) / y_test)) * 100
r2_nn = r2_score(y_test, y_pred_nn)
 
print(f"RMSE:  {rmse_nn:.4f}")
print(f"MAE:   {mae_nn:.4f}")
print(f"MAPE:  {mape_nn:.4f} %")
print(f"R2:    {r2_nn:.4f}")
 
# Usporedba metrika linearne regresije i neuronske mreze
print("\n" + "=" * 60)
print("Usporedba metrika - Linearna regresija vs Neuronska mreza")
print("=" * 60)
print(f"{'Metrika':<10} {'Lin. regresija':>18} {'Neuronska mreza':>18}")
print(f"{'RMSE':<10} {rmse_lr:>18.4f} {rmse_nn:>18.4f}")
print(f"{'MAE':<10} {mae_lr:>18.4f} {mae_nn:>18.4f}")
print(f"{'MAPE':<10} {mape_lr:>18.4f} {mape_nn:>18.4f}")
print(f"{'R2':<10} {r2_lr:>18.4f} {r2_nn:>18.4f}")