def grade_category(grade):
    if grade < 0.0 or grade > 1.0:
        print('Neispravna ocjena. Unesite vrijednost između 0.0 i 1.0.')
        return
    
    if grade >= 0.9:
        print('A')
    elif grade >= 0.8:
        print('B')
    elif grade >= 0.7:
        print('C')
    elif grade >= 0.6:
        print('D')
    else:
        print('F')

try:
    grade = float(input('Unesite ocjenu (0.0-1.0): '))
    grade_category(grade)
    
except ValueError:
    print('Greška. Niste unijeli broj.')