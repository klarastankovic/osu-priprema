import numpy as np

a = np.array([3,1,5], float)
b = np.array([2,4,8], float)
print(a+b)
print(a-b)
print(a*b)
print(a/b)

print(a.min())
print(a.argmin())
print(a.max())
print(a.argmax())
print(a.sum())
print(a.mean())

print(np.mean(a))
print(np.max(a))
print(np.sum(a))

a.sort()
print(a)