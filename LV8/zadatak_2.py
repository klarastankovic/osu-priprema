import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.models import load_model


model = load_model("FCN.keras")
model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

wrong_indices = np.where(y_pred_classes != y_test)[0]
print(f"Broj pogrešno klasificiranih primjera: {len(wrong_indices)}")

fig, axes = plt.subplots(2, 5)
for i, ax in enumerate(axes.flat):
    idx = wrong_indices[i]
    ax.imshow(x_test[idx], cmap='gray')
    ax.set_title(f'Stvarna: {y_test[idx]}\nPredviđena: {y_pred_classes[idx]}')
    ax.axis('off')
plt.suptitle('Pogrešno klasificirane slike iz testnog skupa')
plt.tight_layout()
plt.show()