import json
from flask import Flask, request
from gensim.models import KeyedVectors
from flask import jsonify
import argparse
import sys
import socket
import time
import logging

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
port = 8888
localIp = "localhost"


def isNoneWords(word):
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
    if isNoneWords(word):
        return jsonify("word is null or not in model!")
    else:
        return jsonify({'word': word, 'vector': model.word_vec(word).tolist()})


@app.route("/sim", methods=['GET'])
def similarity_route():
    word1 = request.args.get("word1")
    word2 = request.args.get("word2")
    if isNoneWords(word1) or isNoneWords(word2):
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
    if isNoneWords(word):
        return jsonify("word is null or not in model!")
    else:
        return jsonify({
            'word':
            word,
            'top_similar_words':
            model.similar_by_word(word, topn=20, restrict_vocab=None)
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
