import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.models import load_model


model = load_model("FCN.keras")
model.summary()

photo = plt.imread("test.png")

if photo.ndim == 3:
    photo = np.mean(photo[:, :, :3], axis=2)
 
if photo.max() > 1.0:
    photo = photo.astype("float32") / 255
else:
    photo = photo.astype("float32")
 
if photo.mean() > 0.5:
    photo = 1.0 - photo
 
photo_resized = np.array(keras.preprocessing.image.smart_resize(
    np.expand_dims(photo, axis=-1), (28, 28)
))
 
photo_input = np.expand_dims(photo_resized, axis=0)

photo_pred = model.predict(photo_input)
photo_pred_class = np.argmax(photo_pred)
photo_confidence = photo_pred[0][photo_pred_class] * 100
print(f"Predviđena klasa: {photo_pred_class} s pouzdanosti {photo_confidence:.2f}%")

plt.figure()
plt.imshow(photo_resized[:, :, 0], cmap='gray')
plt.title(f'Predviđeno: {photo_pred_class}')
plt.axis('off')
plt.show()


fig, axes = plt.subplots(2, 5)
for i, ax in enumerate(axes.flat):
    photo = plt.imread(f"test{i}.png")

    if photo.ndim == 3:
        photo = np.mean(photo[:, :, :3], axis=2)

    if photo.max() > 1.0:
        photo = photo.astype("float32") / 255
    else:
        photo = photo.astype("float32")

    if photo.mean() > 0.5:
        photo = 1.0 - photo

    photo_resized = np.array(keras.preprocessing.image.smart_resize(
        np.expand_dims(photo, axis=-1), (28, 28)
    ))
    photo_input = np.expand_dims(photo_resized, axis=0)

    photo_pred = model.predict(photo_input)
    photo_pred_class = np.argmax(photo_pred)
    photo_confidence = photo_pred[0][photo_pred_class] * 100
    print(f"Stvarno: {i}, Predviđena klasa: {photo_pred_class} s pouzdanosti {photo_confidence:.2f}%")

    ax.imshow(photo_resized, cmap='gray')
    ax.set_title(f'Predviđeno: {photo_pred_class}')
    ax.axis('off')
plt.tight_layout()
plt.show()