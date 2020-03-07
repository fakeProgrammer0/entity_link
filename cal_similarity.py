import gensim.models.keyedvectors as KeyedVectors
import gensim.matutils as matutils
import numpy as np

# embedding_filepath = './data/Tencent_AILab_ChineseEmbedding.txt'
# embedding_filepath = './data/Tencent_AILab_ChineseEmbedding_Min.txt'
embedding_filepath = './data/top10k.txt'


class SimilarityMeasure:

    def __init__(self, label, similarity_callback):
        self.label = label
        self.similarity_callback = similarity_callback


def get_similarity_measures():

    def L1(A, x):
        return 1 / np.linalg.norm(A - x, 1, axis=1)

    def L2(A, x):
        return 1 / np.sqrt(np.linalg.norm(A - x, 1, axis=1))

    measures = [
        SimilarityMeasure('cos', np.dot),
        SimilarityMeasure('L1', L1),
        SimilarityMeasure('L2', L2)
    ]
    return measures


def is_none_words(word, wv):
    if word is None or len(word) == 0 or word not in wv.vocab:
        return True
    else:
        return False


def most_similar_words(wv, word, measures, top_n=10):
    if is_none_words(word, wv):
        return {}

    result = {}
    vector = wv.get_vector(word)
    word_index = wv.vocab[word].index
    wv.init_sims()
    for measure in measures:
        dists = measure.similarity_callback(wv.vectors_norm, vector)
        best = matutils.argsort(dists, topn=top_n + 1, reverse=True)
        top_words = [(wv.index2word[sim], float(dists[sim])) for sim in best if sim != word_index]
        result[measure.label] = top_words[:top_n]

    return result


def print_similar_words(similar_words):
    for label, word_score_pairs in similar_words.items():
        print(label, end=' : ')
        for word_score in word_score_pairs:
            print("{} : {:.4f}".format(*word_score), end=' ; ')
        print(end='\n')


def ouput_similar_words(word, similar_words, fname=None):
    if fname is None:
        fname = word + "_top_similar_words.txt"

    with open(fname, 'w', encoding='utf8') as out_file:
        out_file.write('word : {}\n'.format(word))
        for label, word_score_pairs in similar_words.items():
            out_file.write(label + ' : ')
            for word_score in word_score_pairs:
                out_file.write("{} : {:.4f} ; ".format(*word_score))
            out_file.write('\n')


if __name__ == '__main__':
    word_vectors = KeyedVectors.Word2VecKeyedVectors.load_word2vec_format(embedding_filepath,
                                                                          binary=False)
    word = input('word: ')
    while word != " ":
        top_words = most_similar_words(word_vectors, word, get_similarity_measures())
        print_similar_words(top_words)
        ouput_similar_words(word, top_words)
        word = input('\nword: ')

    print('Done!')
