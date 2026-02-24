import statistics


def collect_numbers():
    numbers = []
    
    while True:
        user_input = input('Unesite broj (ili "Done" za završetak): ')
        if user_input == 'Done':
            break
        
        try:
            value = int(user_input)    
            numbers.append(value)
            
        except ValueError:
            print('Neispravan unos. Unesite cijeli broj ili "Done" za završetak.')
            
    return numbers


def calculate_statistics(numbers):
    if not numbers:
        print('Nema unesenih brojeva.')
        return
    
    average = statistics.mean(numbers)
    minimum = min(numbers)
    maximum = max(numbers)
    numbers.sort()
    
    print(f'Uneseno {len(numbers)} brojeva.')
    print(f'Srednja vrijednost: {average:.2f}')
    print(f'Minimum: {minimum}')
    print(f'Maksimum: {maximum}')
    print(f'Sortirana lista brojeva: {numbers}')
  
  
def main():
    numbers = collect_numbers()
    calculate_statistics(numbers)
   
    
if __name__ == '__main__':
    main()