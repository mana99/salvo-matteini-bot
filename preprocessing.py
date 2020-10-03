import json

import numpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from os.path import join, dirname

from keras_preprocessing.text import Tokenizer

from model import WORD_EMBEDDING_SIZE, logger, INPUT_DIR

MAX_NWORDS_QUANTILE = 0.99
INPUT_DIR = join(dirname(__file__), 'shared')
TOKENIZER_PATH = join(INPUT_DIR, 'tokenizer.pickle')

START_TOKEN, END_TOKEN = 0, -1
# non usare -1 (errore in embedding layer)


def preprocess(tweets_df):

    # todo remove characters

    # todo entities (e.g. https://www.debian.org/ --> \URL) (see utils.tweet_parsing)

    # remove tweets with too many words
    tweets_df['full_text_nwords'] = tweets_df.apply(lambda x: len(x['full_text'].split()), axis=1)
    max_words = int(tweets_df['full_text_nwords'].quantile(MAX_NWORDS_QUANTILE))
    tweets = list(tweets_df[tweets_df['full_text_nwords'] <= max_words]['full_text'])

    # encode
    t = Tokenizer()
    t.fit_on_texts(tweets)
    vocab_size = len(t.word_index) + 1   # todo: constant
    encoded_tweets = t.texts_to_sequences(tweets)

    # # pad documents
    # proc_encoded_tweets = pad_sequences(encoded_tweets, maxlen=max_words, padding='post')

    proc_encoded_tweets = [[START_TOKEN] + tweet for tweet in encoded_tweets]

    return proc_encoded_tweets, t, vocab_size, max_words


# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# http://www.italianlp.it/resources/italian-word-embeddings/

def get_t128_italiannlp_embedding(tokenizer: Tokenizer, vocab_size: int) -> np.array:

    # t128 size: 1188949, 1027699 (lower)

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, WORD_EMBEDDING_SIZE))

    # load the whole embedding into memory
    # takes a while... (~1-2 min)
    logger.info("Loading pre-trained word embedding in memory (~1-2 mins)...")
    with open(join(INPUT_DIR, 'twitter128.json'), 'r') as fin:
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