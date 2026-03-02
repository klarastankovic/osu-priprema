import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

# a) Na koliko osoba su izvrsena mjerenja?
broj_osoba = data.shape[0]
print(f"Mjerenja su izvrsena na {broj_osoba} osoba.")

# b) Scatter plot - odnos visine i mase
plt.figure()
plt.scatter(data[:, 1], data[:, 2], s=5, color='blue', alpha=0.4)
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.title('Odnos visine i mase osobe')

# c) Scatter plot - odnos visine i mase (svaka 50. osoba)
plt.figure()
plt.scatter(data[::50, 1], data[::50, 2], s=20, color='green', alpha=0.8)
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.title('Odnos visine i mase - svaka 50. osoba')

# d) Min, max i srednja vrijednost visine
print(f'\nMin visina: {data[:, 1].min():.2f} cm')
print(f'Max visina: {data[:, 1].max():.2f} cm')
print(f'Srednja visina: {data[:, 1].mean():.2f} cm')

# e) Min, max i srednja vrijednost visine (m/z)
ind_m  = (data[:, 0] == 1.0)
ind_z = (data[:, 0] == 0.0)

print('\nMuskarci:')
print(f'Min visina: {data[ind_m, 1].min():.2f} cm')
print(f'Max visina: {data[ind_m, 1].max():.2f} cm')
print(f'Srednja visina: {data[ind_m, 1].mean():.2f} cm')

print(f'\nZene:')
print(f"Min visina: {data[ind_z, 1].min():.2f} cm")
print(f"Max visina: {data[ind_z, 1].max():.2f} cm")
print(f"Srednja visina: {data[ind_z, 1].mean():.2f} cm")

plt.show()