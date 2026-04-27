import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print(df.head(5))
print(df.info())
print(df["target"])
print("Klase:", data.target_names)
print("Raspodjela:\n", df["target"].value_counts(sort=False))

plt.scatter(df["mean radius"], df["mean texture"], c=df["target"])
plt.xlabel("mean radius")
plt.ylabel("mean texture")
plt.title("Raspodjela klasa")
plt.colorbar(label="0=maligno, 1=benigno")
plt.show()




