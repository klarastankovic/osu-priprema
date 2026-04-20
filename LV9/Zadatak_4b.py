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


# Premala stopa učenja
print("\nLearning rate = 0.0001")
model = build_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [keras.callbacks.TensorBoard(log_dir='logs/lr_0001', update_freq=100)]
model.fit(X_train_n, y_train, epochs=15, batch_size=64,
          callbacks=callbacks, validation_split=0.1)
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost (lr=0.0001): {100.0*score[1]:.2f}%')

# Optimalna stopa učenja (Adam default)
print("\nLearning rate = 0.001")
model = build_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [keras.callbacks.TensorBoard(log_dir='logs/lr_001', update_freq=100)]
model.fit(X_train_n, y_train, epochs=15, batch_size=64,
          callbacks=callbacks, validation_split=0.1)
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost (lr=0.001): {100.0*score[1]:.2f}%')

# Prevelika stopa učenja
print("\nLearning rate = 0.1")
model = build_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1),
              loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [keras.callbacks.TensorBoard(log_dir='logs/lr_01', update_freq=100)]
model.fit(X_train_n, y_train, epochs=15, batch_size=64,
          callbacks=callbacks, validation_split=0.1)
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost (lr=0.1): {100.0*score[1]:.2f}%')