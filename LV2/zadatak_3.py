import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:, :, 0].copy()

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Originalna slika")

# a) Posvijetliti sliku
brightened_img = np.clip(img + 50, 0, 255).astype(np.uint8)
axes[1].imshow(brightened_img, cmap="gray")
axes[1].set_title('Posvijetljena slika')

# b) Prikazati samo drugu cetvrtinu slike po sirini
width = img.shape[1]
second_quarter_img = img[:, width // 4 : width // 2]
axes[2].imshow(second_quarter_img, cmap="gray")
axes[2].set_title('Druga cetvrtina po sirini')

# c) Zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
rotated_img = np.rot90(img, k=-1)
axes[3].imshow(rotated_img, cmap="gray")
axes[3].set_title('Rotacija 90°')

# d) Zrcaliti sliku
mirrored_img = np.fliplr(img)
axes[4].imshow(mirrored_img, cmap="gray")
axes[4].set_title('Zrcaljena slika')

plt.tight_layout()
plt.show()