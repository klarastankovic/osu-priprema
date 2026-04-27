import pandas as pd

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print(df.head(5))
print(df.info())
print(df["target"])
