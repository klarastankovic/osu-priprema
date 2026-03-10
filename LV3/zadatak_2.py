import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

# a)
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20, edgecolor='white')
plt.title('Histogram Emisije CO2 (g/km)')
plt.xlabel('CO2 emisija (g/km)')
plt.ylabel('Broj vozila')


# b)
plt.figure()
for ft, label in {'X': 'Regularni benzin', 'Z': 'Premium benzin',
                   'D': 'Dizel', 'E': 'Etanol (E85)', 'N': 'Prirodni plin'}.items():
    grp = data[data['Fuel Type'] == ft]
    plt.scatter(grp['Fuel Consumption City (L/100km)'],
                grp['CO2 Emissions (g/km)'],
                label=label, s=20, alpha=0.5)
plt.xlabel('Gradska potrošnja (L/100km)')
plt.ylabel('CO2 Emisija (g/km)')
plt.title('Gradska potrošnja vs CO2 emisija')
plt.legend(title='Tip goriva')


# c)
data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.suptitle('')
plt.title('Izvangradska potrošnja po tipu goriva')
plt.xlabel('Tip goriva')
plt.ylabel('Izvangradska potrošnja (L/100km)')


# d)
plt.figure()
data.groupby('Fuel Type', observed=True).size().plot(kind='bar')
plt.title('Broj vozila po tipu goriva')
plt.xlabel('Tip goriva')
plt.ylabel('Broj vozila')
plt.xticks(rotation=0)


# e)
plt.figure()
data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().plot(kind='bar')
plt.title('Prosječna CO2 emisija po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna CO2 emisija (g/km)')
plt.xticks(rotation=0)


plt.show()