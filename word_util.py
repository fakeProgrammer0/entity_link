import sys
import pickle

embedding_filepath = './data/'
words_filepath = './data/words.txt'
word_indices_filepath = './data/word_indices.pickle'

def extract_words(input_filename=embedding_filepath, output_filename=words_filepath):
    with open(input_filename, mode='r', encoding="utf8") as in_file:
        with open(output_filename, mode='w', encoding="utf8") as out_file:
            for line in in_file:
                line = line[:line.find(' ')]
                if line != '':
                    out_file.write(line + '\n')

def get_word_indices():
    with open(words_filepath, 'r', encoding='utf8') as in_file:
        indices = dict()
        for (i, line) in enumerate(in_file):
            indices[i + 1] = line[:line.find('\n')]
        return indices

def build_word_indices():
    with open(words_filepath, 'r', encoding='utf8') as in_file:
        indices = dict()
        for (i, line) in enumerate(in_file):
            indices[i + 1] = line[:line.find('\n')]
        with open(word_indices_filepath, 'wb') as out_file:
            pickle.dump(indices, out_file)

def load_word_indices():
    return pickle.load(word_indices_filepath)

if __name__ == "__main__":
    build_word_indices()
    indices = load_word_indices()
    print(indices[100])