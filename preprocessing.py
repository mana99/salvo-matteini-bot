import itertools
import logging
import json
import string

import numpy as np
import pandas as pd
import preprocessor
import re

from os.path import join, dirname
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import unicode
import nltk

logger = logging.getLogger(__name__)

# todo env variables
INPUT_DIR = join(dirname(__file__), 'shared')
EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.json')

MAX_NWORDS_QUANTILE = 0.99
MAX_NWORDS = 52
START_TOKEN = 0  # non usare -1 (errore in embedding layer)


def preprocess(tweets):

    m = tweets

    # filter
    # max_words = int(pd.Series(map(lambda x: len(x.split()), m)).quantile(MAX_NWORDS_QUANTILE))
    m = filter(lambda x: len(x.split()) <= MAX_NWORDS, m)

    # lowercase
    m = map(str.lower, m)

    # remove punct
    # punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    punct = """!"#$%&()*+-;<=>?[\]^`{|}~"""
    table = str.maketrans('', '', punct)
    m = map(lambda x: x.translate(table), m)

    # normalize_text
    # m = map(lambda x: normalize_text(x.split()), m)
    # m = map(normalize_text, m)
    m = map(lambda sentence: ' '.join([normalize_text(word) for word in sentence.split()]), m)

    # tweet-preprocessor
    m = map(tweet_parsing, m)

    # remove punct (2)
    punct2 = ':.,@'  # _
    table2 = str.maketrans('', '', punct2)
    m = map(lambda x: x.translate(table2), m)

    # tokenize
    # m = map(tokenize, m)
    # t.tokenize(sentence)

    # encode


    m, m2 = itertools.tee(m)
    t = nltk.RegexpTokenizer(r"[dnl]['´`]|\w+|\$[\d\.]+|\S+")
    tokenizer = Tokenizer(t)
    list(map(lambda x: tokenizer.fit_on_texts(t.tokenize(x)), m))
    m = map(lambda x: [START_TOKEN]+[tokenizer.word_index.get(w) for w in t.tokenize(x) if tokenizer.word_index.get(w)], m2)

    return m, tokenizer



def get_digits(text):
    try:
        val = int(text)
    except:
        text = re.sub('\d', '@Dg', text)
        # text = re.sub('\d', '\\\Dg', text)
        return text
    if val >= 0 and val < 2100:
        return str(val)
    else:
        return "DIGLEN_" + str(len(str(val)))
        # return "\\DIGLEN" + str(len(str(val)))


def normalize_text(word):
    if "http" in word or ("." in word and "/" in word):
        word = unicode("___URL___")
        # word = '\\URL'
        return word
    if len(word) > 26:
        return "__LONG-LONG__"
        # return "\\LONGLONG"
    new_word = get_digits(word)
    if new_word != word:
        word = new_word
    if word[0].isupper():
        word = word.capitalize()
    else:
        word = word.lower()
    return word



def tokenize(sentence):
    # try:
    #     tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')
    # except LookupError:
    #     nltk.download('punkt')
    #     tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')
    t = nltk.RegexpTokenizer(r"[dnl]['´`]|\w+|\$[\d\.]+|\S+")
    return t.tokenize(sentence)


def filter_tweets(tweets):
    # remove tweets with too many words
    tweets['full_text_nwords'] = tweets.apply(lambda x: len(x['full_text'].split()), axis=1)
    max_words = int(tweets['full_text_nwords'].quantile(MAX_NWORDS_QUANTILE))
    return list(tweets[tweets['full_text_nwords'] <= max_words]['full_text']), max_words


def encode_tweets(tweets):

    # encode
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    encoded_tweets = tokenizer.texts_to_sequences(tweets)

    # # pad documents
    # proc_encoded_tweets = pad_sequences(encoded_tweets, maxlen=max_words, padding='post')

    encoded_tweets = [[START_TOKEN] + tweet for tweet in encoded_tweets]

    return encoded_tweets, tokenizer


# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# http://www.italianlp.it/resources/italian-word-embeddings/

def get_t128_italiannlp_embedding(tokenizer: Tokenizer, n_words: int) -> np.array:

    # t128 size: 1188949, 1027699 (lowercase)

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((n_words, 128))
    # todo: constant size

    # load the whole embedding into memory
    # takes a while... (~1-2 min)
    logger.info("Loading pre-trained word embedding in memory (~1-2 mins)...")
    with open(EMBEDDING_PATH, 'r') as fin:
        t128 = json.load(fin)

    logger.info("Building embedding matrix...")
    for word, i in tokenizer.word_index.items():
        embedding_matrix[i] = t128.get(word, list(np.random.choice([1, -1]) * np.random.rand(128+1)))[:-1]

    # sql_engine = create_engine(f"sqlite:///{WORD_EMBEDDING_PATH}")
    # connection = sql_engine.raw_connection()
    # for word, i in tokenizer.word_index.items():
    #     res = t128[t128['key_lower'] == word.lower()]  # troppo lento
    #     res = pd.read_sql(sql=f'select * from store where key = "{word}"', con=connection)
    #     if len(res) == 1:
    #         embedding_matrix[i] = res.drop(['key', 'ranking'], axis=1).values[0]

    return embedding_matrix




EMOJI_PREFIX = '/EMOJI'
HASHTAG_PREFIX = '/HASHTAG'
URL_PREFIX = '/URL'
MENTIONS_PREFIX = '/MENTION'


def tweet_parsing(tweet):

    # create the parser
    parser_tweet = preprocessor.parse(tweet)

    # check and substitute for the element we do not want
    if parser_tweet.emojis:
        for emoji in parser_tweet.emojis:
            emoji_str = emoji.match
            tweet = tweet.replace(emoji_str, '')

    if parser_tweet.hashtags:
        for hashtag in parser_tweet.hashtags:
            hashtag_str = hashtag.match
            tweet = tweet.replace(hashtag_str, hashtag_str[1:])

    # if parser_tweet.urls:
    #     for url in parser_tweet.urls:
    #         url_str = url.match
    #         tweet = tweet.replace(url_str, URL_PREFIX)

    if parser_tweet.mentions:
        for mention in parser_tweet.mentions:
            mention_str = mention.match
            tweet = tweet.replace(mention_str, '')

    # Preprocessor is /HASHTAG#awesome 👍 /URLhttps://github.com/s/preprocessor
    return tweet


if __name__ == '__main__':
    print(get_digits("15"))
    print(get_digits("2099"))
    print(get_digits("2100"))
    print(get_digits("Ho visto 3000 manigoldi"))
    print(normalize_text("aNtoNio"))
    print(normalize_text("cane"))
    print(normalize_text("CANE"))
    print(normalize_text("Ho visto 3000 manigoldi"))
    print(normalize_text("100,23"))
    s = "Questo è l'apostrofo, con una virgola ed un punto e virgola;"
    t = nltk.data.load('tokenizers/punkt/italian.pickle')
    print(t.tokenize(s))
    t2 = nltk.RegexpTokenizer(r"[dnl]['´`]|\w+|\$[\d\.]+|\S+")
    print(t2.tokenize(s))
