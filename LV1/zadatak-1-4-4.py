def extract_text_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read().lower()

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return ""


def count_words(text):
    if not text:
        return {}
    
    words = text.split()
    word_counts = {}
    
    for word in words:
        clean_word = ''.join(char for char in word if char.isalnum())
        if clean_word:
            word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
    
    return word_counts


def extract_unique_words(word_counts):
    unique_words = [word for word, count in word_counts.items() if count == 1]
    return unique_words


def main():
    filename = 'song.txt'
    text = extract_text_from_file(filename)
    
    if text:
        word_counts = count_words(text)
        unique_words = extract_unique_words(word_counts)
        
        print(f'Riječi koje se pojavljuju samo jednom ({len(unique_words)}): {unique_words}')
    

if __name__ == '__main__':
    main()