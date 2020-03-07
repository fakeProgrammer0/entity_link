import json
from flask import Flask, request
from gensim.models import KeyedVectors
from flask import jsonify
import argparse
import sys
import socket
import time
import logging
import gensim.matutils as matutils
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s;%(levelname)s: %(message)s",
                              "%Y-%m-%d %H:%M:%S")
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(console)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

model = None
port = 28888
localIp = "localhost"


def is_none_words(word):
    if word is None or len(word) == 0 or word not in model.vocab:
        return True
    else:
        return False


@app.route("/", methods=['GET'])
def welcome():

    vecAPI = "http://" + localIp + ":" + str(port) + "/vec?word=淘宝"
    simAPI = "http://" + localIp + ":" + str(port) + "/sim?word1=淘宝&word2=京东"
    topSimAPI = "http://" + localIp + ":" + str(port) + "/top_sim?word=淘宝"

    return "Welcome to word2vec api . <br/>\
    try this api below：<br/> \
    1. vec api:    <a href='" + vecAPI + "'>" + vecAPI + "</a> <br/>\
    2. sim api:    <a href='" + simAPI + "'>" + simAPI + "</a> <br/>\
    3. top sim api:    <a href='" + topSimAPI + "'>" + topSimAPI + "</a> <br/>\
    "


@app.route("/vec", methods=['GET'])
def vec_route():
    word = request.args.get("word")
    if is_none_words(word):
        return jsonify("word is null or not in model!")
    else:
        return jsonify({'word': word, 'vector': model.word_vec(word).tolist()})


@app.route("/sim", methods=['GET'])
def similarity_route():
    word1 = request.args.get("word1")
    word2 = request.args.get("word2")
    if is_none_words(word1) or is_none_words(word2):
        return jsonify("word is null or not in model!")
    else:
        return jsonify({
            'word1': word1,
            'word2': word2,
            'similarity': float(model.similarity(word1, word2))
        })


@app.route("/top_sim", methods=['GET'])
def top_similarity_route():
    word = request.args.get("word")
    if is_none_words(word):
        return jsonify("word is null or not in model!")
    else:
        return jsonify({
            'word':
            word,
            'top_similar_words':
            model.similar_by_word(word, topn=20, restrict_vocab=None)
        })

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


def most_similar_words(word, measures, top_n=10):
    if is_none_words(word):
        return {}

    result = {}
    vector = model.get_vector(word)
    word_index = model.vocab[word].index
    model.init_sims()
    for measure in measures:
        dists = measure.similarity_callback(model.vectors_norm, vector)
        best = matutils.argsort(dists, topn=top_n + 1, reverse=True)
        top_words = [(model.index2word[sim], float(dists[sim])) for sim in best if sim != word_index]
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


predefine_measures = get_similarity_measures()


@app.route("/top_words", methods=['GET'])
def top_similarity_words_route():
    word = request.args.get("word")
    if is_none_words(word):
        return jsonify("word is null or not in model!")
    else:
        top_words = most_similar_words(word, predefine_measures)
        ouput_similar_words(word, top_words)
        return jsonify({
            'word':
            word,
            'top_similar_words':
            top_words
        })


def getLocalIP():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def main():
    global model
    global port
    global localIp
    for arg in sys.argv[1:]:
        logger.debug(arg)
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Path to the trained model")
    p.add_argument("--host", help="Host name (default: localhost)")
    p.add_argument("--port", help="Port (default: 8888)")
    args = p.parse_args()
    host = args.host if args.host else "localhost"
    port = int(args.port) if args.port else 8888
    localIp = getLocalIP()
    if not args.model:
        logger.debug(
            "Usage: w2v.py --model model_path [--host host --port 8888]")
        sys.exit(1)
    logger.debug("start load model:" + str(args.model))
    start_time = time.time()
    model = KeyedVectors.load_word2vec_format(args.model, binary=False)
    logger.debug("end load model:" + str(args.model))
    # app.run(host=host, port=port,debug=True)
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
