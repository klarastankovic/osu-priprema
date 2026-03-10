import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

print('a)')
print(f'Broj mjerenja: {data.shape[0]}')
print(f'Broj stupaca (veličina): {data.shape[1]}')

print('\nTipovi podataka:')
print(data.dtypes)

print('\nProvjera izostalih ili dupliciranih vrijednosti:')
missing = data.isnull().sum()
print(f'Ukupan broj izostalih vrijednosti: {missing.sum()}')
duplicates = data.duplicated().sum()
print(f'Ukupan broj dupliciranih redaka: {duplicates}')

if missing.sum() > 0:
    print("\nBrišem retke s izostalim vrijednostima...")
    data.dropna(axis=0)
if duplicates > 0:
    print("Brišem duplicirane retke...")
    data.drop_duplicates()
data.reset_index(drop=True)

categories = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
print('\nKonverzija kategoričkih veličina')
for category in categories:
    data[category] = data[category].astype('category')
    print(
        f'Broj jedinstvenih vrijednosti u stupcu "{category}": {data[category].nunique()}')

print('\nTipovi podataka nakon konverzije:')
print(data.dtypes)


print('\n\nb)')
sorted_cities = data.sort_values(by='Fuel Consumption City (L/100km)')
print('3 automobila s najvećom gradskom potrošnjom:')
print(sorted_cities[['Make', 'Model',
      'Fuel Consumption City (L/100km)']].tail(3))
print('\n3 automobila s najmanjom gradskom potrošnjom:')
print(sorted_cities[['Make', 'Model',
      'Fuel Consumption City (L/100km)']].head(3))


print('\n\nc)')
mask_engine = (data['Engine Size (L)'] >= 2.5) & (
    data['Engine Size (L)'] <= 3.5)
filtered_data = data[mask_engine]
print(
    f'Broj automobila s motorom između 2.5L i 3.5L: {filtered_data.shape[0]}')
print(
    f'Procječna CO2 emisija za ova vozila: {filtered_data["CO2 Emissions (g/km)"].mean():.2f} g/km')


print('\n\nd)')
audi = data[data['Make'] == 'Audi']
print(f'Broj mjerila za Audi: {audi.shape[0]}')
print(
    f'Prosječna CO2 emisija Audi vozila sa 4 cilindra: {audi[audi["Cylinders"] == 4]["CO2 Emissions (g/km)"].mean():.2f} g/km')


print('\n\ne)')
cylinders = data.groupby('Cylinders')
print('Broj vozila po broju cilindara (>4):')
for cyl, count in cylinders.size().items():
    if cyl >= 4:
        print(f'{cyl}: {count}')

print('Prosječna CO2 emisija po broju cilindara (>4):')
for cyl, co2 in cylinders['CO2 Emissions (g/km)'].mean().items():
    print(f'{cyl}: {co2:.2f} g/km')
    
    
print('\n\nf)')
diesel = data[data['Fuel Type'] == 'D']
regular = data[data['Fuel Type'] == 'X']
print('Broj dizel vozila: ', diesel.shape[0])
print('Broj benzinskih vozila: ', regular.shape[0])

print('Prosječna gradska potrošnja')
print(f'Dizel: {diesel["Fuel Consumption City (L/100km)"].mean():.2f} L/100km')
print(f'Benzin: {regular["Fuel Consumption City (L/100km)"].mean():.2f} L/100km')

print('Medijalna gradska potrošnja')
print(f'Dizel: {diesel["Fuel Consumption City (L/100km)"].median():.2f} L/100km')
print(f'Benzin: {regular["Fuel Consumption City (L/100km)"].median():.2f} L/100km')


print('\n\ng)')
diesel_4cyl = diesel[diesel['Cylinders'] == 4]
worst_diesel = diesel_4cyl.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).iloc[0]
print('Dizel vozilo s 4 cilindra i najvećom gradskom potrošnjom:')
print(worst_diesel[['Make', 'Model', 'Fuel Consumption City (L/100km)']])


print('\n\nh)')
manual = data[data['Transmission'].str.startswith('M')]
print(f'Broj vozila s ručnim mjenjačem: {manual.shape[0]}')


print('\n\ni)')
corr = data.corr(numeric_only=True)
print('Korelacija između numeričkih veličina:')
print(corr.round(3).to_string())