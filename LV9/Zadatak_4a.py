import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train) #, dtype ="uint8"
y_test = to_categorical(y_test) #, dtype ="uint8"


def build_model():
    model = keras.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))
    model.add(layers.Conv2D(32,  (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,  (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))
    return model


# Mala veličina serije
print("\nBatch size = 8")
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [keras.callbacks.TensorBoard(log_dir='logs/batch_8', update_freq=100)]
model.fit(X_train_n, y_train, epochs=15, batch_size=8,
          callbacks=callbacks, validation_split=0.1)
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost (batch=8): {100.0*score[1]:.2f}%')

# Srednja veličina serije
print("\nBatch size = 64")
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [keras.callbacks.TensorBoard(log_dir='logs/batch_64', update_freq=100)]
model.fit(X_train_n, y_train, epochs=15, batch_size=64,
          callbacks=callbacks, validation_split=0.1)
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost (batch=64): {100.0*score[1]:.2f}%')

# Velika veličina serije
print("\nBatch size = 512")
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [keras.callbacks.TensorBoard(log_dir='logs/batch_512', update_freq=100)]
model.fit(X_train_n, y_train, epochs=15, batch_size=512,
          callbacks=callbacks, validation_split=0.1)
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost (batch=512): {100.0*score[1]:.2f}%')