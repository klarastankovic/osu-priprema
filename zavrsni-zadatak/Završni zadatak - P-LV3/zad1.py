import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(f"type(X_train): {type(X_train)}")
print(f"type(X_test): {type(X_test)}")
print(f"type(y_train): {type(y_train)}")
print(f"type(y_test): {type(y_test)}")

plt.imshow(X_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()