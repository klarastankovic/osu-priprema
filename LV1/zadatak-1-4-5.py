import statistics


def extract_sms_data(filename):
    data = []
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    data.append((parts[0], parts[1].strip()))
        return data
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []


def average_words(data, label_type):
    word_counts = [len(message.split()) for label, message in data if label == label_type]
    
    if not word_counts:
        return 0

    return statistics.mean(word_counts)


def count_spam_exclamations(data):
    count = sum(1 for label, message in data if label == 'spam' and message.endswith('!'))
    return count


def main():
    filename = 'SMSSpamCollection.txt'
    data = extract_sms_data(filename)
    
    if data:
        average_ham = average_words(data, 'ham')
        average_spam = average_words(data, 'spam')
        spam_exclamations = count_spam_exclamations(data)
        
        print(f'Prosječan broj riječi (ham): {average_ham:.2f}')
        print(f'Prosječan broj riječi (spam): {average_spam:.2f}')
        print(f'Spam koje završavaju s "!": {spam_exclamations}')


if __name__ == '__main__':
    main()