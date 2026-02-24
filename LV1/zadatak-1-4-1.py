def total_euro(hours, rate):
    return hours * rate

hours = float(input('Radni sati: '))
rate = float(input('eura/h: '))

total = total_euro(hours, rate)
print(f'Ukupno: {total} eura')