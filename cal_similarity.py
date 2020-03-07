import gensim.models.keyedvectors as KeyedVectors
import numpy as np

# embedding_filepath = './data/Tencent_AILab_ChineseEmbedding.txt'
# embedding_filepath = './data/Tencent_AILab_ChineseEmbedding_Min.txt'
embedding_filepath = './data/top10k.txt'

word_vectors = KeyedVectors.Word2VecKeyedVectors.load_word2vec_format(embedding_filepath, binary=False)


if __name__ == '__main__':
    print(word_vectors.similar_by_word('时间'))
    print('Done!')




