import numpy as np

np.random.seed(56) #postavi seed generatora brojeva
rNumbers = np.random.rand(10) #generiraj 10 slucajnih brojeva
print(rNumbers)
print(rNumbers.mean())