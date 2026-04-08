import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
#plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


# broj boja na slici (original)
n_unique = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
print(f"Broj razlicitih boja u originalnoj slici: {n_unique}")

# K-means
km = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=0)
km.fit(img_array)

# zamjeni RGB vrijednosti elemenata slike sa RGB vrijednostima centara klastera kojem element pripada
labels = km.predict(img_array)
img_array_aprox = km.cluster_centers_[labels]

# kvantizirana slika
img_kvant = np.reshape(img_array_aprox, (w,h,d))

plt.figure()
plt.title("Kvantizirana slika")
plt.imshow(img_kvant)
plt.tight_layout()


# primjena na ostale slike
for i in range(2, 7):
    img_i = Image.imread(f"imgs\\test_{i}.jpg")
    
    # provjera za RGBA
    if img_i.dtype == np.uint8:
        img_i = img_i.astype(np.float64) / 255
    else:
        img_i = img_i.astype(np.float64)
        
    if img_i.shape[2] == 4:
        img_i = img_i[:, :, :3]
    
    wi, hi, di = img_i.shape
    img_array_i = img_i.reshape(wi*hi, di)
    img_array_aprox_i = img_array_i.copy()
    
    km_i = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
    km_i.fit(img_array_i)
    labels_i = km_i.predict(img_array_i)
    img_array_aprox_i = km_i.cluster_centers_[labels_i]
    img_i_kvant = np.reshape(img_array_aprox_i, (wi, hi, di))
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_i); ax[0].set_title(f"Originalna slika"); ax[0].axis('off')
    ax[1].imshow(img_i_kvant); ax[1].set_title(f"Kvantizirana slika"); ax[1].axis('off')
    plt.tight_layout()


# metoda lakta
K_range = range(1, 11)
inertias = []
for k in K_range:
    km_temp = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=0)
    km_temp.fit(img_array)
    inertias.append(km_temp.inertia_)
 
plt.figure()
plt.plot(list(K_range), inertias, 'o-')
plt.xlabel('Broj grupa K')
plt.ylabel('Kriterijska funkcija J')
plt.title('Lakat metoda – test_1.jpg')
plt.xticks(list(K_range))


# binarna slika
km_bin = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
km_bin.fit(img_array)
labels_bin = km_bin.predict(img_array)

fig, axes = plt.subplots(1, 5)
for k in range(5):
    mask = (labels_bin == k).reshape(w, h).astype(np.uint8)
    axes[k].imshow(mask, cmap='gray')
    axes[k].set_title(f'Grupa {k+1}')
    axes[k].axis('off')
plt.suptitle(f'Binarne maske – test_1.jpg')
plt.tight_layout()

plt.show()