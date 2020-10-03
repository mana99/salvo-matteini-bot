
import logging
import json
import numpy as np

from os.path import join, dirname
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


logger = logging.getLogger(__name__)

# todo env variables
INPUT_DIR = join(dirname(__file__), 'shared')
EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.json')

MAX_NWORDS_QUANTILE = 0.99
START_TOKEN = 0  # non usare -1 (errore in embedding layer)


def filter_tweets(tweets):
    # remove tweets with too many words
    tweets['full_text_nwords'] = tweets.apply(lambda x: len(x['full_text'].split()), axis=1)
    max_words = int(tweets['full_text_nwords'].quantile(MAX_NWORDS_QUANTILE))
    return list(tweets[tweets['full_text_nwords'] <= max_words]['full_text']), max_words


def encode_tweets(tweets):

    # todo remove characters

    # todo entities (e.g. https://www.debian.org/ --> \URL) (see utils.tweet_parsing)

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
        embedding_matrix[i] = t128.get(word, list(np.random.choice([1, -1]) * np.random.rand(WORD_EMBEDDING_SIZE+1)))[:-1]

    # sql_engine = create_engine(f"sqlite:///{WORD_EMBEDDING_PATH}")
    # connection = sql_engine.raw_connection()
    # for word, i in tokenizer.word_index.items():
    #     res = t128[t128['key_lower'] == word.lower()]  # troppo lento
    #     res = pd.read_sql(sql=f'select * from store where key = "{word}"', con=connection)
    #     if len(res) == 1:
    #         embedding_matrix[i] = res.drop(['key', 'ranking'], axis=1).values[0]

    return embedding_matrix