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


# 50% skupa za učenje
n_total   = X_train_n.shape[0]
n_reduced = n_total // 2
idx = np.random.choice(n_total, n_reduced, replace=False)

X_train_small = X_train_n[idx]
y_train_small = y_train[idx]

print(f"\nSmanjen skup za učenje: {n_reduced} uzoraka")

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

model.summary()

callbacks = [keras.callbacks.TensorBoard(log_dir='logs/mali_skup', update_freq=100)]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_small, y_train_small, epochs=20, batch_size=64,
          callbacks=callbacks, validation_split=0.1)

score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost (50% skupa): {100.0*score[1]:.2f}%')